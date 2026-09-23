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


def assign_mvp_roles(metrics: pd.DataFrame) -> pd.DataFrame:
    """Простые, временные эвристики MVP; они будут заменены на следующем этапе."""
    result = metrics.copy()
    # Для узла без входящих средств отношение не определено; в сдаваемой CSV
    # используем 0, чтобы не оставлять обязательное поле пустым.
    result["pass_through"] = result.pass_through.fillna(0.0)
    result["role"] = "peripheral"
    result.loc[(result.out_deg == 0) & ~result.truncated_by_depth, "role"] = "terminal"
    result.loc[(result.in_deg > 0) & (result.out_deg > 0), "role"] = "transit"
    result.loc[(result.in_deg == 0) & (result.out_deg > 0), "role"] = "distributor"
    result.loc[result.is_seed & (result.out_deg > 0), "role"] = "coordinator"
    result["role_score"] = np.select(
        [result.role.eq("peripheral"), result.role.eq("terminal")], [0.50, 0.60], default=0.55
    )

    # Один технический кластер на MVP: это гарантирует согласованность двух выгрузок.
    result["cluster_id"] = 0
    turnover = result.in_kzt + result.out_kzt
    max_turnover = turnover.max()
    result["priority_score"] = (
        turnover / max_turnover if max_turnover > 0 else pd.Series(0.0, index=result.index)
    )
    result["priority_score"] = result.priority_score.fillna(0.0).clip(0.0, 1.0)
    result["pagerank"] = 0.0  # Поле схемы; вычисление PageRank намеренно отложено.
    result["evidence"] = result.apply(
        lambda row: (
            f"in_deg={row.in_deg}, out_deg={row.out_deg}, "
            f"in_kzt={row.in_kzt:.0f}, out_kzt={row.out_kzt:.0f}, depth={row.depth}"
        ),
        axis=1,
    )
    return result


def write_outputs(features: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_roles = features[OUTPUT_COLUMNS["nodes_roles"]].copy()
    nodes_roles.to_csv(out_dir / "nodes_roles.csv", index=False)

    ordered = features.sort_values(["priority_score", "gid"], ascending=[False, True])
    top_gids = ",".join(map(str, ordered.head(10).gid))
    clusters = pd.DataFrame([{
        "cluster_id": 0,
        "n_nodes": len(features),
        "n_seed": int(features.is_seed.sum()),
        "sum_kzt_internal": float(features.out_kzt.sum()),
        "top_gids": top_gids,
        "hypothesis": "MVP: единый технический кластер без содержательной интерпретации",
    }], columns=OUTPUT_COLUMNS["clusters"])
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
    features = assign_mvp_roles(basic_metrics(graph, nodes))
    write_outputs(features, args.out)
    print(f"Загружено: {len(nodes)} узлов, {len(edges)} рёбер, {len(transactions)} транзакций")
    print(f"Граф: {graph.number_of_nodes()} узлов, {graph.number_of_edges()} рёбер")
    print(f"MVP-выгрузки записаны в: {args.out}")


if __name__ == "__main__":
    main()
