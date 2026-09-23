#!/usr/bin/env python3
"""MVP решения кейса «Граф денег».

Запуск:
    python solution.py --data data --out out
"""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


OUTPUT_COLUMNS = {
    "nodes_roles": [
        "gid", "role", "role_score", "cluster_id", "priority_score", "evidence",
        "in_deg", "out_deg", "in_kzt", "out_kzt", "pagerank", "pass_through",
        "depth", "is_seed", "truncated_by_depth",
    ],
    "clusters": [
        "cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis",
    ],
    "top_nodes": ["rank", "gid", "role", "priority_score", "why"],
}


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает входные таблицы и делает минимальные проверки схемы."""
    paths = {name: data_dir / f"{name}.parquet" for name in ("edges", "nodes", "transactions")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Не найдены входные файлы: " + ", ".join(missing))

    edges = pd.read_parquet(paths["edges"])
    nodes = pd.read_parquet(paths["nodes"])
    transactions = pd.read_parquet(paths["transactions"])
    transactions["date"] = pd.to_datetime(transactions["date"])

    expected = {
        "edges": {"src", "dst", "sum_kzt", "n_tx", "depth"},
        "nodes": {"gid", "depth", "is_seed"},
        "transactions": {"src", "dst", "date", "sum_kzt"},
    }
    actual = {"edges": edges, "nodes": nodes, "transactions": transactions}
    for name, columns in expected.items():
        absent = columns - set(actual[name].columns)
        if absent:
            raise ValueError(f"В {name}.parquet отсутствуют колонки: {sorted(absent)}")
    if nodes.gid.duplicated().any():
        raise ValueError("gid должен быть уникальным в nodes.parquet")

    return edges, nodes, transactions


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    """Строит направленный граф из уже агрегированных рёбер."""
    graph = nx.DiGraph()
    for edge in edges.itertuples(index=False):
        graph.add_edge(
            edge.src,
            edge.dst,
            sum_kzt=float(edge.sum_kzt),
            n_tx=int(edge.n_tx),
            depth=int(edge.depth),
        )
    return graph


def basic_metrics(graph: nx.DiGraph, nodes: pd.DataFrame) -> pd.DataFrame:
    """Считает только степени и обороты: без центральностей и кластеризации."""
    result = nodes[["gid", "depth", "is_seed"]].copy()
    result["in_deg"] = result.gid.map(dict(graph.in_degree())).fillna(0).astype(int)
    result["out_deg"] = result.gid.map(dict(graph.out_degree())).fillna(0).astype(int)
    result["in_kzt"] = result.gid.map(dict(graph.in_degree(weight="sum_kzt"))).fillna(0.0)
    result["out_kzt"] = result.gid.map(dict(graph.out_degree(weight="sum_kzt"))).fillna(0.0)
    result["in_tx"] = result.gid.map(dict(graph.in_degree(weight="n_tx"))).fillna(0).astype(int)
    result["out_tx"] = result.gid.map(dict(graph.out_degree(weight="n_tx"))).fillna(0).astype(int)
    result["pass_through"] = np.where(
        result.in_kzt > 0, result.out_kzt / result.in_kzt, np.nan
    )
    result["truncated_by_depth"] = (result.depth == 4) & (result.out_deg == 0)
    return result


def weighted_pagerank(
    graph: nx.DiGraph, alpha: float = 0.85, tolerance: float = 1e-10, max_iter: int = 200
) -> dict:
    """Взвешенный PageRank без дополнительной зависимости scipy.

    Вес ``sum_kzt`` определяет долю влияния, передаваемую по исходящим рёбрам.
    Узлы без рёбер не входят в граф и получают PageRank 0 при объединении с nodes.
    """
    gids = list(graph.nodes)
    n_nodes = len(gids)
    if n_nodes == 0:
        return {}

    index = {gid: index for index, gid in enumerate(gids)}
    scores = np.full(n_nodes, 1.0 / n_nodes)
    outgoing = []
    dangling = []
    for gid in gids:
        edges = list(graph.out_edges(gid, data="sum_kzt"))
        total_weight = sum(float(weight) for _, _, weight in edges)
        if total_weight == 0:
            dangling.append(index[gid])
            outgoing.append(())
        else:
            outgoing.append(tuple(
                (index[target], float(weight) / total_weight)
                for _, target, weight in edges
            ))

    for _ in range(max_iter):
        next_scores = np.full(n_nodes, (1.0 - alpha) / n_nodes)
        next_scores += alpha * scores[dangling].sum() / n_nodes
        for source_index, targets in enumerate(outgoing):
            for target_index, share in targets:
                next_scores[target_index] += alpha * scores[source_index] * share
        if np.abs(next_scores - scores).sum() < tolerance:
            return dict(zip(gids, next_scores))
        scores = next_scores
    raise RuntimeError("PageRank не сошёлся за заданное число итераций")


def weighted_hits(
    graph: nx.DiGraph, tolerance: float = 1e-10, max_iter: int = 300
) -> tuple[dict, dict]:
    """Взвешенный HITS: hub для распределителей, authority для сборщиков."""
    gids = list(graph.nodes)
    n_nodes = len(gids)
    if n_nodes == 0:
        return {}, {}

    index = {gid: index for index, gid in enumerate(gids)}
    edges = [
        (index[src], index[dst], float(data["sum_kzt"]))
        for src, dst, data in graph.edges(data=True)
    ]
    hubs = np.full(n_nodes, 1.0 / np.sqrt(n_nodes))
    authorities = np.zeros(n_nodes)
    for _ in range(max_iter):
        next_authorities = np.zeros(n_nodes)
        for source, target, weight in edges:
            next_authorities[target] += hubs[source] * weight
        authority_norm = np.linalg.norm(next_authorities)
        if authority_norm > 0:
            next_authorities /= authority_norm

        next_hubs = np.zeros(n_nodes)
        for source, target, weight in edges:
            next_hubs[source] += next_authorities[target] * weight
        hub_norm = np.linalg.norm(next_hubs)
        if hub_norm > 0:
            next_hubs /= hub_norm

        delta = np.abs(next_hubs - hubs).sum() + np.abs(next_authorities - authorities).sum()
        hubs, authorities = next_hubs, next_authorities
        if delta < tolerance:
            return dict(zip(gids, hubs)), dict(zip(gids, authorities))
    raise RuntimeError("HITS не сошёлся за заданное число итераций")


def capped_log_score(values: pd.Series, quantile: float = 0.99) -> pd.Series:
    """Нормирует положительную величину в 0..1, ограничивая влияние выбросов."""
    cap = float(values.quantile(quantile))
    if cap <= 0:
        return pd.Series(0.0, index=values.index)
    return np.log1p(values.clip(lower=0, upper=cap)) / np.log1p(cap)


def temporal_features(transactions: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Считает активность и скорость дальнейшего перевода средств по датам."""
    tx = transactions.reset_index(names="_tx_id").copy()
    activity = pd.concat([
        tx[["_tx_id", "src", "date", "sum_kzt"]].rename(columns={"src": "gid"}),
        tx[["_tx_id", "dst", "date", "sum_kzt"]].rename(columns={"dst": "gid"}),
    ], ignore_index=True).drop_duplicates(["_tx_id", "gid"])
    summary = activity.groupby("gid").agg(
        active_days=("date", "nunique"),
        tx_count_total=("_tx_id", "size"),
        avg_tx_amount=("sum_kzt", "mean"),
    ).reset_index()

    result = nodes[["gid", "is_seed"]].merge(summary, on="gid", how="left")
    result[["active_days", "tx_count_total", "avg_tx_amount"]] = result[
        ["active_days", "tx_count_total", "avg_tx_amount"]
    ].fillna(0.0)
    result["active_days"] = result.active_days.astype(int)
    result["tx_count_total"] = result.tx_count_total.astype(int)
    frequency = result.tx_count_total / result.active_days.replace(0, np.nan)
    result["frequency_score"] = capped_log_score(frequency.fillna(0.0), quantile=0.95)

    incoming_dates = tx.groupby("dst").date.apply(lambda dates: np.sort(dates.values))
    outgoing_dates = tx.groupby("src").date.apply(lambda dates: np.sort(dates.values))
    timing_by_gid = {}
    for gid in set(incoming_dates.index) & set(outgoing_dates.index):
        incoming = incoming_dates[gid]
        outgoing = outgoing_dates[gid]
        next_outgoing = np.searchsorted(outgoing, incoming, side="left")
        valid = next_outgoing < len(outgoing)
        if valid.any():
            lag_days = (outgoing[next_outgoing[valid]] - incoming[valid]).astype("timedelta64[D]").astype(int)
            timing_by_gid[gid] = max(0.0, 1.0 - float(lag_days.min()) / 7.0)

    result["transit_timing_score"] = result.gid.map(timing_by_gid).fillna(0.0)
    # Для seed входящие средства не полностью представлены в 4-hop выборке.
    result.loc[result.is_seed, "transit_timing_score"] = 0.0
    return result.drop(columns="is_seed")


def graph_structural_features(graph: nx.DiGraph, nodes: pd.DataFrame) -> pd.DataFrame:
    """Нормирует HITS и directed betweenness в дополнительные признаки 0..1."""
    hubs, authorities = weighted_hits(graph)
    shortest_path_graph = graph.copy()
    for _, _, data in shortest_path_graph.edges(data=True):
        # Крупный денежный поток интерпретируется как более сильная связь.
        data["distance"] = 1.0 / float(data["sum_kzt"])
    betweenness = nx.betweenness_centrality(
        shortest_path_graph, weight="distance", normalized=True
    )

    result = nodes[["gid"]].copy()
    result["hub_score"] = result.gid.map(hubs).fillna(0.0)
    result["authority_score"] = result.gid.map(authorities).fillna(0.0)
    result["betweenness_score"] = result.gid.map(betweenness).fillna(0.0)
    for column in ("hub_score", "authority_score", "betweenness_score"):
        result[column] = capped_log_score(result[column])
    return result


def add_graph_role_support(features: pd.DataFrame) -> pd.DataFrame:
    """Подтверждает уже назначенную роль структурным сигналом, не меняя label."""
    result = features.copy()
    result["_role_graph_support"] = 0.0
    result.loc[result.role.eq("distributor"), "_role_graph_support"] = result.hub_score
    result.loc[result.role.eq("consolidator"), "_role_graph_support"] = result.authority_score
    result.loc[result.role.eq("transit"), "_role_graph_support"] = result.betweenness_score
    result["role_score"] = (result.role_score + 0.05 * result._role_graph_support).clip(0.0, 1.0)
    return result


def role_thresholds(metrics: pd.DataFrame) -> dict[str, float]:
    """Пороги ролей из текущей выгрузки, а не вручную заданные константы."""
    outgoing = metrics.loc[metrics.out_deg > 0]
    return {
        # Сборщик: верхние 5% по числу входящих контрагентов и переводов.
        "high_in_deg": float(metrics.in_deg.quantile(0.95)),
        "high_in_tx": float(metrics.in_tx.quantile(0.95)),
        # Распределитель: верхние 10% среди фактически отправляющих узлов.
        "high_out_deg": float(outgoing.out_deg.quantile(0.90)),
        "high_out_tx": float(outgoing.out_tx.quantile(0.90)),
        "high_out_kzt": float(outgoing.out_kzt.quantile(0.90)),
        "seed_out_deg": float(outgoing.out_deg.quantile(0.75)),
        "seed_out_tx": float(outgoing.out_tx.quantile(0.75)),
    }


def assign_roles(metrics: pd.DataFrame) -> pd.DataFrame:
    """Прозрачные правила ролей на степенях, оборотах и количестве переводов."""
    result = metrics.copy()
    # Для узла без входящих средств отношение не определено; в сдаваемой CSV
    # используем 0, чтобы не оставлять обязательное поле пустым.
    result["pass_through"] = result.pass_through.fillna(0.0)
    thresholds = role_thresholds(result)
    result["role"] = "peripheral"

    # Глубина 4 означает обрезанную выгрузку, поэтому эти узлы всегда остаются
    # peripheral до отдельного анализа: terminal им присваивать нельзя.
    eligible = ~result.truncated_by_depth
    consolidator = (
        eligible
        & (result.in_deg >= thresholds["high_in_deg"])
        & (result.in_tx >= thresholds["high_in_tx"])
        & ((result.out_deg <= 1) | (result.pass_through <= 0.50))
    )
    distributor = (
        eligible
        & (result.out_deg >= thresholds["high_out_deg"])
        & (result.out_tx >= thresholds["high_out_tx"])
        & (result.out_kzt >= thresholds["high_out_kzt"])
    )
    coordinator = (
        eligible
        & result.is_seed
        & (result.out_deg >= thresholds["seed_out_deg"])
        & (result.out_tx >= thresholds["seed_out_tx"])
    )
    transit = (
        eligible
        & (result.in_deg > 0)
        & (result.out_deg > 0)
        & result.pass_through.between(0.60, 1.50)
    )
    terminal = eligible & (result.in_deg > 0) & (result.out_deg == 0)

    # Приоритет правил: широкое распределение > seed-координатор > сборщик >
    # транзит > конечный получатель. Так у каждого gid ровно одна роль.
    result.loc[terminal, "role"] = "terminal"
    result.loc[transit, "role"] = "transit"
    result.loc[consolidator, "role"] = "consolidator"
    result.loc[coordinator, "role"] = "coordinator"
    result.loc[distributor, "role"] = "distributor"

    result["role_score"] = 0.50
    result.loc[result.role.eq("terminal"), "role_score"] = 0.75
    result.loc[result.role.eq("transit"), "role_score"] = (
        0.60 + 0.30 * (1 - (result.loc[result.role.eq("transit"), "pass_through"] - 1).abs())
    ).clip(0.60, 0.90)
    result.loc[result.role.eq("consolidator"), "role_score"] = (
        0.65 + 0.10 * (result.loc[result.role.eq("consolidator"), "in_deg"] / thresholds["high_in_deg"] - 1)
    ).clip(0.65, 0.95)
    result.loc[result.role.eq("coordinator"), "role_score"] = (
        0.65 + 0.10 * (result.loc[result.role.eq("coordinator"), "out_deg"] / thresholds["seed_out_deg"] - 1)
    ).clip(0.65, 0.95)
    result.loc[result.role.eq("distributor"), "role_score"] = (
        0.65 + 0.10 * (result.loc[result.role.eq("distributor"), "out_deg"] / thresholds["high_out_deg"] - 1)
    ).clip(0.65, 0.95)

    def evidence(row: pd.Series) -> str:
        if row.role == "consolidator":
            return (f"in_deg={row.in_deg}, in_tx={row.in_tx}, in_kzt={row.in_kzt:.0f}, "
                    f"out_deg={row.out_deg}, pass_through={row.pass_through:.2f}")
        if row.role == "distributor":
            return (f"out_deg={row.out_deg}, out_tx={row.out_tx}, out_kzt={row.out_kzt:.0f}, "
                    f"in_deg={row.in_deg}")
        if row.role == "coordinator":
            return (f"is_seed=1, out_deg={row.out_deg}, out_tx={row.out_tx}, "
                    f"out_kzt={row.out_kzt:.0f}")
        if row.role == "transit":
            return (f"in_kzt={row.in_kzt:.0f}, out_kzt={row.out_kzt:.0f}, "
                    f"pass_through={row.pass_through:.2f}, in_tx={row.in_tx}, out_tx={row.out_tx}")
        if row.role == "terminal":
            return f"in_kzt={row.in_kzt:.0f}, in_tx={row.in_tx}, out_deg=0, depth={row.depth}"
        return (f"in_deg={row.in_deg}, out_deg={row.out_deg}, depth={row.depth}, "
                f"truncated_by_depth={int(row.truncated_by_depth)}")

    result["evidence"] = result.apply(evidence, axis=1)
    return result


def assign_clusters(graph: nx.DiGraph, features: pd.DataFrame) -> pd.DataFrame:
    """Выделяет сообщества на неориентированной проекции денежного графа."""
    projection = nx.Graph()
    projection.add_nodes_from(features.gid)
    for src, dst, data in graph.edges(data=True):
        weight = float(data["sum_kzt"])
        if projection.has_edge(src, dst):
            projection[src][dst]["sum_kzt"] += weight
        else:
            projection.add_edge(src, dst, sum_kzt=weight)

    communities = nx.community.louvain_communities(
        projection, weight="sum_kzt", seed=42
    )
    # Стабильные номера: сначала крупные сообщества, затем по минимальному gid.
    communities = sorted(
        communities,
        key=lambda community: (-len(community), min(map(str, community))),
    )
    cluster_by_gid = {
        gid: cluster_id
        for cluster_id, community in enumerate(communities)
        for gid in community
    }
    result = features.copy()
    result["cluster_id"] = result.gid.map(cluster_by_gid).astype(int)
    return result


def cluster_statistics(graph: nx.DiGraph, features: pd.DataFrame) -> pd.DataFrame:
    """Считает размер и направленный внутренний оборот каждого кластера."""
    cluster_by_gid = features.set_index("gid").cluster_id.to_dict()
    internal_kzt = {cluster_id: 0.0 for cluster_id in features.cluster_id.unique()}
    for src, dst, data in graph.edges(data=True):
        src_cluster = cluster_by_gid[src]
        if src_cluster == cluster_by_gid[dst]:
            internal_kzt[src_cluster] += float(data["sum_kzt"])

    stats = features.groupby("cluster_id", sort=True).agg(
        n_nodes=("gid", "size"), n_seed=("is_seed", "sum")
    ).reset_index()
    stats["sum_kzt_internal"] = stats.cluster_id.map(internal_kzt).astype(float)
    return stats


def add_priority_scores(graph: nx.DiGraph, features: pd.DataFrame) -> pd.DataFrame:
    """Добавляет объяснимый приоритет для очереди AML-проверки.

    priority_score = 95% предыдущих бизнес-факторов + 5% структурный сигнал,
    где структурный сигнал = HITS hub, HITS authority и betweenness.
    """
    result = features.copy()
    turnover = result.in_kzt + result.out_kzt
    result["_turnover_score"] = capped_log_score(turnover)

    role_weight = {
        "consolidator": 0.90,
        "distributor": 0.85,
        "coordinator": 0.80,
        "transit": 0.75,
        "terminal": 0.45,
        "peripheral": 0.20,
    }
    result["_role_score"] = result.role.map(role_weight).astype(float)

    flow_score = pd.Series(0.0, index=result.index)
    flow_nodes = (result.in_kzt > 0) & (result.out_kzt > 0) & ~result.is_seed
    flow_score.loc[flow_nodes] = (
        np.minimum(result.loc[flow_nodes, "in_kzt"], result.loc[flow_nodes, "out_kzt"])
        / np.maximum(result.loc[flow_nodes, "in_kzt"], result.loc[flow_nodes, "out_kzt"])
    )
    result["_flow_score"] = flow_score

    pagerank = weighted_pagerank(graph)
    result["pagerank"] = result.gid.map(pagerank).fillna(0.0)
    pagerank_cap = float(result.pagerank.quantile(0.99))
    result["_pagerank_score"] = (
        result.pagerank / pagerank_cap if pagerank_cap > 0 else 0.0
    )
    result["_pagerank_score"] = result._pagerank_score.clip(0.0, 1.0)

    # depth показывает расстояние от seed в исходной 4-hop выгрузке.
    result["_seed_depth_score"] = (1.0 - result.depth / 4.0).clip(0.0, 1.0)
    result.loc[result.is_seed, "_seed_depth_score"] = 1.0

    cluster_stats = cluster_statistics(graph, result).set_index("cluster_id")
    cluster_turnover = result.cluster_id.map(cluster_stats.sum_kzt_internal)
    cluster_size = result.cluster_id.map(cluster_stats.n_nodes)
    result["_cluster_score"] = (
        0.70 * capped_log_score(cluster_turnover)
        + 0.30 * capped_log_score(cluster_size)
    )
    result["_temporal_score"] = (
        0.60 * result.frequency_score + 0.40 * result.transit_timing_score
    )
    result["_structural_score"] = (
        0.34 * result.hub_score
        + 0.33 * result.authority_score
        + 0.33 * result.betweenness_score
    )

    base_priority = (
        0.28 * result._turnover_score
        + 0.19 * result._role_score
        + 0.14 * result._flow_score
        + 0.14 * result._pagerank_score
        + 0.09 * result._seed_depth_score
        + 0.09 * result._cluster_score
        + 0.07 * result._temporal_score
    )
    result["priority_score"] = (0.95 * base_priority + 0.05 * result._structural_score).clip(0.0, 1.0)

    result["evidence"] = result.apply(
        lambda row: (
            f"{row.evidence} | p:t={row._turnover_score:.2f},r={row._role_score:.2f},"
            f"f={row._flow_score:.2f},pr={row._pagerank_score:.2f},"
            f"sd={row._seed_depth_score:.2f},c={row._cluster_score:.2f},"
            f"tm={row._temporal_score:.2f},g={row._structural_score:.2f} | "
            f"tm:d={row.active_days},"
            f"tx={row.tx_count_total},av={row.avg_tx_amount:.0f},"
            f"fr={row.frequency_score:.2f},ti={row.transit_timing_score:.2f} | "
            f"g={row.hub_score:.2f}/{row.authority_score:.2f}/{row.betweenness_score:.2f}"
        ),
        axis=1,
    )
    return result


def cluster_hypothesis(cluster: pd.DataFrame) -> str:
    """Формулирует короткую проверяемую гипотезу по составу сообщества."""
    counts = cluster.role.value_counts()
    n_nodes = len(cluster)
    n_distributors = int(counts.get("distributor", 0))
    n_coordinators = int(counts.get("coordinator", 0))
    n_consolidators = int(counts.get("consolidator", 0))
    n_transit = int(counts.get("transit", 0))
    n_terminal = int(counts.get("terminal", 0))

    if n_nodes == 1 and cluster.iloc[0].in_deg == 0 and cluster.iloc[0].out_deg == 0:
        return "Изолированный узел: in_deg=0, out_deg=0"
    if n_distributors + n_coordinators > 0:
        return (f"Контур распределения: distributors={n_distributors}, "
                f"coordinators={n_coordinators}, nodes={n_nodes}")
    if n_consolidators > 0:
        return f"Контур сбора: consolidators={n_consolidators}, nodes={n_nodes}"
    if n_transit >= max(2, n_nodes // 3):
        return f"Транзитный контур: transit={n_transit}, nodes={n_nodes}"
    if n_terminal >= max(1, n_nodes // 2):
        return f"Ветка конечных получателей: terminal={n_terminal}, nodes={n_nodes}"
    return f"Смешанный денежный контур: transit={n_transit}, terminal={n_terminal}, nodes={n_nodes}"


def build_cluster_summary(graph: nx.DiGraph, features: pd.DataFrame) -> pd.DataFrame:
    """Собирает обязательную строку clusters.csv для каждого сообщества."""
    rows = []
    stats = cluster_statistics(graph, features).set_index("cluster_id")

    for cluster_id, cluster in features.groupby("cluster_id", sort=True):
        top = cluster.sort_values(
            ["priority_score", "gid"], ascending=[False, True]
        ).head(10)
        rows.append({
            "cluster_id": int(cluster_id),
            "n_nodes": len(cluster),
            "n_seed": int(cluster.is_seed.sum()),
            "sum_kzt_internal": float(stats.loc[cluster_id, "sum_kzt_internal"]),
            "top_gids": ",".join(map(str, top.gid)),
            "hypothesis": cluster_hypothesis(cluster),
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS["clusters"])


def write_outputs(features: pd.DataFrame, graph: nx.DiGraph, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_roles = features[OUTPUT_COLUMNS["nodes_roles"]].copy()
    nodes_roles.to_csv(out_dir / "nodes_roles.csv", index=False)

    ordered = features.sort_values(["priority_score", "gid"], ascending=[False, True])
    clusters = build_cluster_summary(graph, features)
    clusters.to_csv(out_dir / "clusters.csv", index=False)

    top_count = min(20, len(ordered))
    top_nodes = ordered.head(top_count).reset_index(drop=True)
    top_nodes = pd.DataFrame({
        "rank": np.arange(1, top_count + 1),
        "gid": top_nodes.gid,
        "role": top_nodes.role,
        "priority_score": top_nodes.priority_score,
        "why": top_nodes.evidence,
    })
    top_nodes.to_csv(out_dir / "top_nodes.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP анализа графа денежных переводов")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    edges, nodes, transactions = load_data(args.data)
    graph = build_graph(edges)
    features = assign_roles(basic_metrics(graph, nodes))
    features = assign_clusters(graph, features)
    features = features.merge(temporal_features(transactions, nodes), on="gid", how="left")
    features = features.merge(graph_structural_features(graph, nodes), on="gid", how="left")
    features = add_graph_role_support(features)
    features = add_priority_scores(graph, features)
    write_outputs(features, graph, args.out)
    print(f"Загружено: {len(nodes)} узлов, {len(edges)} рёбер, {len(transactions)} транзакций")
    print(f"Граф: {graph.number_of_nodes()} узлов, {graph.number_of_edges()} рёбер")
    print(f"Выгрузки записаны в: {args.out}")


if __name__ == "__main__":
    main()
