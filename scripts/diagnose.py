"""Run with: uv run streamlit run scripts/diagnose.py"""

from html import escape
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt
import pandas as pd
import streamlit as st
import torch

from nyt_connections_solver.checkpoints import load_graph_harness
from nyt_connections_solver.evaluation import load_puzzles
from nyt_connections_solver.evaluation.diagnostics import DiagnosticSession, summarize_record


GROUP_COLORS = {"Purple": "#9b8ac4", "Blue": "#78a8d2", "Green": "#8ebf87", "Yellow": "#dfc364"}


@st.cache_data(show_spinner=False)
def checkpoint_metadata(path, modified):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return {key: value for key, value in checkpoint.items() if key != "model_state_dict"}


@st.cache_data(show_spinner=False)
def puzzles_for_split(dataset, modified, dates):
    """Keep historical answer row order for Purple/Blue/Green/Yellow labels."""
    return load_puzzles(Path(dataset), dates)


def display_record(record, *, key):
    state = record.final_state
    status = "Solved" if state.won else "Out of mistakes" if state.finished else "In progress"
    st.caption(f"Game status: {status} · {len(state.solved_groups)}/4 groups accepted · "
               f"{state.mistakes_remaining} incorrect guesses left. "
               f"Run settings: beam width {record.beam_width}, "
               f"allowed incorrect guesses {record.max_mistakes}, shuffle seed {record.shuffle_seed}.")
    if not record.decisions:
        return
    index = st.select_slider(
        "Replay / inspect guess (display only)", options=list(range(len(record.decisions))),
        value=len(record.decisions) - 1,
        format_func=lambda i: f"Guess {i + 1}" + (" (pending)" if record.decisions[i].feedback is None else ""),
        key=f"decision_{key}_{len(record.decisions)}",
        help="Select a recorded decision. This does not change the guess limit, rerun search, or submit a guess.",
    )
    decision = record.decisions[index]
    before = decision.before
    cols = st.columns(4)
    cols[0].metric("Inspecting", f"Guess {index + 1}")
    cols[1].metric("Accepted before this guess", f"{len(before.solved_groups)}/4")
    cols[2].metric("Incorrect guesses left before this guess", before.mistakes_remaining)
    cols[3].metric("Feature construction", f"{record.feature_seconds:.2f}s")
    outcome = decision.feedback.outcome.value if decision.feedback else "not submitted"
    st.markdown(f"**Harness guess:** {' · '.join(decision.chosen)} — **{outcome}**")
    st.caption(f"Search: {decision.search_seconds:.3f}s · Diagnostic ranking/pruning: "
               f"{decision.diagnostic_seconds:.3f}s. Diagnostics run after the policy chooses.")
    show_gold = st.checkbox("Color words by ground truth", value=True, key=f"gold_{key}")
    gold_lookup = {word: i for i, group in enumerate(record.gold_groups) for word in group}
    gold_labels = record.gold_group_labels or tuple(f"Group {i + 1}" for i in range(4))
    colors = [GROUP_COLORS.get(label, "#bbbbbb") for label in gold_labels]
    tiles = []
    for word in record.words:
        color = colors[gold_lookup[word]] if show_gold else "#dddddd"
        border = "3px solid #a84400" if word in decision.chosen else "1px solid #888"
        opacity = "1" if word in before.remaining else "0.3"
        tiles.append(f'<div style="background:{color};border:{border};opacity:{opacity};'
                     f'padding:12px;text-align:center;color:#111;border-radius:6px">{escape(word)}</div>')
    st.markdown('<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px">'
                + ''.join(tiles) + '</div>', unsafe_allow_html=True)
    st.caption("Orange border: selected guess. Faded: already accepted. "
               "Historical CSV row order determines difficulty colors: Purple, Blue, Green, Yellow.")

    rankings = pd.DataFrame(decision.rankings)
    true_rows = rankings[rankings.true_group]
    chosen_row = rankings[rankings.chosen].iloc[0]
    cols = st.columns(4)
    cols[0].metric("Chosen raw rank", int(chosen_row.raw_rank))
    cols[1].metric("Best true group rank", int(true_rows.raw_rank.min()))
    cols[2].metric("Model mass on true groups", f"{true_rows.model_probability.sum():.2%}")
    cols[3].metric("Candidates pruned", f"{(~rankings.feasible).sum()}/{len(rankings)}")
    rank_tab, plan_tab, projection_tab, history_tab = st.tabs([
        "Candidate rankings", "Search paths", "Word projections & edges", "Ground truth & history",
    ])
    with rank_tab:
        count = st.slider("Candidates to display (view only)", 5, 100, 20, 5, key=f"top_{key}",
                          help="Only changes this table and chart. It does not change beam width or search.")
        feasible_only = st.checkbox("Only feedback-consistent candidates", value=False, key=f"feasible_{key}")
        filtered = rankings[rankings.feasible] if feasible_only else rankings
        # Always include the true groups and chosen guess, even below the cutoff.
        displayed = pd.concat([filtered.head(count), rankings[rankings.true_group | rankings.chosen]])
        displayed = displayed.drop_duplicates("raw_rank").sort_values("raw_rank")
        columns = ["raw_rank", "feasible_rank", "words", "model_probability", "probability_after_pruning",
                   "true_group", "max_gold_overlap", "chosen", "pruning"]
        st.dataframe(displayed[columns], hide_index=True, width="stretch")
        st.caption("Raw rank uses model softmax over every four-word choice. After-pruning probability "
                   "renormalizes scores over all feasible choices before beam truncation, and is used by search. "
                   "Neither is a calibrated chance of correctness. "
                   "True groups and the selected guess are always included in the table.")
        chart_rows = displayed.assign(label=displayed.raw_rank.astype(str) + ": " + displayed.words,
                                      kind=displayed.true_group.map({True: "True group", False: "Other"}))
        chart = alt.Chart(chart_rows).mark_bar().encode(
            x=alt.X("model_probability:Q", title="Model softmax mass"),
            y=alt.Y("label:N", sort=None, title=None),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["True group", "Other"], range=["#468a55", "#89939e"])),
            tooltip=["words", "raw_rank", "model_probability", "pruning"],
        ).properties(height=max(200, 22 * len(chart_rows)))
        st.altair_chart(chart, width="stretch")
        st.download_button("Download all candidate rankings", rankings.drop(columns="group").to_csv(index=False),
                           file_name=f"rankings_{record.date}_guess{index + 1}.csv", key=f"download_{key}_{index}")
    with plan_tab:
        st.markdown("**Projected remaining groups**")
        st.dataframe(pd.DataFrame([
            {"order": i + 1, "words": " · ".join(group),
             "true_group": frozenset(group) in set(map(frozenset, record.gold_groups)),
             "max_gold_overlap": max(len(set(group) & set(gold)) for gold in record.gold_groups)}
            for i, group in enumerate(decision.plan)
        ]), hide_index=True, width="stretch")
        depth = st.selectbox("Search depth to inspect (view only)", list(range(1, len(decision.trace.beams_by_depth) + 1)), key=f"depth_{key}_{index}")
        paths = decision.trace.beams_by_depth[depth - 1]
        st.caption(f"Retained {len(paths)} distinct paths at depth {depth}; configured beam width is {record.beam_width}. "
                   "A wider beam can retain more alternatives without changing the winning guess.")
        st.dataframe(pd.DataFrame([
            {"rank": i + 1, "path": " → ".join(" / ".join(g) for g in path.groups),
             "summed_log_probability": path.log_probability,
             "true_groups_in_path": sum(frozenset(g) in set(map(frozenset, record.gold_groups)) for g in path.groups)}
            for i, path in enumerate(paths)
        ]), hide_index=True, width="stretch")
        st.caption("Paths sum log probabilities recomputed after feedback pruning at every residual state. "
                   "Normalization uses all feasible choices, not just the retained beam. The final group is forced by elimination. "
                   "Every retained path has a feedback-consistent completion. Beam search approximates the best-scoring path.")
    with projection_tab:
        representation = st.selectbox("Representation", list(record.projections), key=f"representation_{key}")
        coordinates, variance = record.projections[representation]
        points = pd.DataFrame({
            "word": record.words, "PC1": coordinates[:, 0], "PC2": coordinates[:, 1],
            "gold_group": [gold_labels[gold_lookup[w]] for w in record.words],
            "status": ["Chosen" if w in decision.chosen else "Remaining" if w in before.remaining else "Accepted" for w in record.words],
        })
        base = alt.Chart(points).encode(x="PC1:Q", y="PC2:Q")
        scatter = base.mark_circle(size=140).encode(
            color=alt.Color("gold_group:N", scale=alt.Scale(domain=list(gold_labels), range=colors)) if show_gold else alt.value("#6c8bac"),
            shape="status:N", opacity=alt.condition(alt.datum.status == "Accepted", alt.value(.25), alt.value(1)),
            tooltip=["word", "gold_group", "status"],
        )
        labels = base.mark_text(dy=-13).encode(text="word:N")
        st.altair_chart((scatter + labels).properties(height=420).interactive(), width="stretch")
        st.caption(f"PCA explains {variance:.1%} of variance in this 16-word board. Coordinates are fixed across guesses. "
                   "Learned graph encoder vectors use the original full board; they are not recomputed for residual states. "
                   "2D distance is a lossy view, not an explanation of a subset score.")
        channel = st.selectbox("Pairwise edge feature", record.edge_names, key=f"edge_{key}")
        values = record.edges[:, :, record.edge_names.index(channel)]
        cells = pd.DataFrame([
            {"first": a, "second": b, "value": float(values[i, j])}
            for i, a in enumerate(record.words) for j, b in enumerate(record.words)
        ])
        heatmap = alt.Chart(cells).mark_rect().encode(
            x=alt.X("first:N", sort=list(record.words), title=None),
            y=alt.Y("second:N", sort=list(record.words), title=None),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="blues", domain=[0, 1])),
            tooltip=["first", "second", "value"],
        ).properties(height=430)
        st.altair_chart(heatmap, width="stretch")
        st.caption("Input edge channel before learned attention projections. This is not an attention-weight map.")
    with history_tab:
        st.dataframe(pd.DataFrame({"group": gold_labels,
                                   "words": [" · ".join(g) for g in record.gold_groups]}), hide_index=True, width="stretch")
        submitted = [d for d in record.decisions if d.feedback is not None]
        st.dataframe(pd.DataFrame([
            {"guess": i + 1, "words": " · ".join(d.chosen), "feedback": d.feedback.outcome.value,
             "search_seconds": d.search_seconds}
            for i, d in enumerate(submitted)
        ]), hide_index=True, width="stretch")


def main():
    st.set_page_config(page_title="Connections model diagnostics", layout="wide")
    # Discard records created before historical colors were preserved. Keep the
    # loaded model so a code reload does not require loading weights again.
    if st.session_state.get("display_schema") != 3:
        for old_key in ("active", "active_key", "batch_records", "batch_context"):
            st.session_state.pop(old_key, None)
        st.session_state.display_schema = 3
    st.title("Connections model diagnostics")
    st.caption("Inspect model rankings, feedback pruning, projected groups, and word representations against historical answers.")
    st.session_state.setdefault("weights_dir", str(ROOT / "weights"))
    st.session_state.setdefault("dataset_path", str(ROOT / "data/connections.csv"))
    weights_dir = Path(st.sidebar.text_input("Weights directory", key="weights_dir")).expanduser()
    paths = sorted(weights_dir.glob("*text_spelling*.pt"), reverse=True)
    if not paths:
        st.info("No text/spelling graph checkpoints found in this directory.")
        return
    weights = st.sidebar.selectbox("Saved model", paths, format_func=lambda p: p.name)
    metadata = checkpoint_metadata(str(weights), weights.stat().st_mtime_ns)
    if metadata.get("model_class") != "GraphSetTransformer" or "feature_metadata" not in metadata:
        st.error("This checkpoint does not contain the supported text/spelling graph model metadata.")
        return
    device = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"])
    dataset = Path(st.sidebar.text_input("Historical dataset", key="dataset_path")).expanduser()
    splits = [name for name in ("val", "test", "train") if metadata.get("split_dates", {}).get(name)]
    if not splits:
        st.error("Checkpoint has no saved split dates.")
        return
    split = st.sidebar.selectbox("Puzzle split", splits)
    st.session_state.setdefault("applied_run_settings", (20, 4, 0))
    st.sidebar.subheader("Run settings")
    with st.sidebar.form("run_settings"):
        draft_beam = st.number_input("Beam width", min_value=1, max_value=100, value=20,
                                    help="Maximum distinct feasible paths kept at each search depth. Wider beams do more work; they may still choose the same guess.")
        draft_mistakes = st.number_input("Allowed incorrect guesses per puzzle", min_value=1, max_value=20, value=4,
                                        help="The game ends after this many incorrect guesses. One-away guesses count; correct guesses do not.")
        draft_seed = st.number_input("Board shuffle seed", min_value=0, value=0)
        apply_settings = st.form_submit_button("Apply run settings")
    if apply_settings:
        st.session_state.applied_run_settings = (draft_beam, draft_mistakes, draft_seed)
        st.session_state.pop("active_key", None)
    beam_width, max_mistakes, seed = st.session_state.applied_run_settings
    st.sidebar.info(f"Applied: beam {beam_width} · {max_mistakes} allowed incorrect guesses · seed {seed}")
    st.sidebar.caption("Apply settings to restart the current puzzle. Completed batch results stay unchanged until you run the batch again.")
    signature = (str(weights.resolve()), weights.stat().st_mtime_ns, device)
    with st.expander("Checkpoint configuration"):
        st.json({"file": weights.name, "model": metadata["model_config"],
                 "training": {k: v for k, v in metadata.get("training_config", {}).items() if k != "training_summary"},
                 "embedding": metadata["embedding_model_name"],
                 "edge_features": metadata["edge_feature_names"]})
    if st.sidebar.button("Load model", type="primary"):
        st.session_state.pop("active", None)
        st.session_state.pop("active_key", None)
        st.session_state.pop("loaded", None)
        st.session_state.pop("batch_records", None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with st.spinner("Loading model and feature providers…"):
            harness = load_graph_harness(weights, device=device, beam_width=beam_width,
                                         cache_dir=ROOT / "data/feature_cache/graph")
        st.session_state.loaded = (signature, harness)
    loaded = st.session_state.get("loaded")
    if loaded is None or loaded[0] != signature:
        st.info("Select a saved model and click Load model. Model loading happens only on request.")
        return
    if not dataset.is_file():
        st.error(f"Dataset not found: {dataset}")
        return
    harness = loaded[1]
    puzzles = puzzles_for_split(str(dataset.resolve()), dataset.stat().st_mtime_ns,
                               tuple(metadata["split_dates"][split]))
    by_date = {p.date: p for p in puzzles}
    dates = sorted(by_date)
    date = st.sidebar.selectbox("Puzzle date", dates)
    config = (beam_width, max_mistakes, seed)
    session_key = (signature, str(dataset.resolve()), dataset.stat().st_mtime_ns, split, date, config)
    if st.session_state.get("active_key") != session_key:
        with st.spinner("Preparing puzzle features…"):
            st.session_state.active = DiagnosticSession(harness, by_date[date], beam_width=beam_width,
                                                        max_mistakes=max_mistakes, seed=seed)
        st.session_state.active_key = session_key
    session = st.session_state.active
    st.subheader(f"{date} · {split}")
    left, middle, right = st.columns(3)
    if left.button("Submit next harness guess", disabled=session.environment.observe().finished, type="primary"):
        with st.spinner("Searching and submitting…"):
            session.advance()
    if middle.button("Run puzzle to completion", disabled=session.environment.observe().finished):
        with st.spinner("Playing puzzle…"):
            session.finish()
    if right.button("Restart puzzle"):
        st.session_state.pop("active_key", None)
        st.rerun()
    with st.spinner("Inspecting next decision…"):
        session.analyze()
    display_record(session.record(), key=f"active_{date}_{config}")

    st.divider()
    st.subheader("Batch evaluation and replay")
    all_puzzles = st.checkbox("Run entire split", value=False)
    selected = dates if all_puzzles else st.multiselect("Puzzles to run", dates, default=dates[:5], key=f"batch_dates_{split}")
    st.info(f"Next batch: {len(selected)} puzzle(s) · beam width {beam_width} · "
            f"{max_mistakes} allowed incorrect guesses per puzzle · shuffle seed {seed}. "
            "Change these in Run settings and click Apply run settings before starting. "
            "The replay slider only selects which recorded guess to display.")
    if st.button("Run selected puzzles", disabled=not selected):
        progress = st.progress(0.0)
        records = {}
        for i, selected_date in enumerate(selected):
            with st.spinner(f"Playing {selected_date} ({i + 1}/{len(selected)})…"):
                run = DiagnosticSession(harness, by_date[selected_date], beam_width=beam_width,
                                        max_mistakes=max_mistakes, seed=seed)
                records[selected_date] = run.finish()
            progress.progress((i + 1) / len(selected))
        st.session_state.batch_records = records
        st.session_state.batch_context = (signature, str(dataset.resolve()), dataset.stat().st_mtime_ns, split, config)
    records = st.session_state.get("batch_records", {})
    context = (signature, str(dataset.resolve()), dataset.stat().st_mtime_ns, split, config)
    if records:
        recorded_context = st.session_state.get("batch_context")
        recorded_beam, recorded_mistakes, recorded_seed = recorded_context[-1]
        st.caption(f"Recorded batch: {Path(recorded_context[0][0]).name} · {recorded_context[3]} split · "
                   f"beam width {recorded_beam} · {recorded_mistakes} allowed incorrect guesses · seed {recorded_seed}.")
        if recorded_context != context:
            st.warning("The applied settings or dataset differ from this recorded batch. "
                       "Run selected puzzles again to recompute results with the settings shown above.")
        summaries = pd.DataFrame([summarize_record(r) for r in records.values()])
        st.metric("Batch solve rate", f"{summaries.won.mean():.1%}")
        st.dataframe(summaries, hide_index=True, width="stretch")
        st.download_button("Download batch summary", summaries.to_csv(index=False), "connections_diagnostics.csv")
        replay_date = st.selectbox("Replay completed puzzle", list(records))
        display_record(records[replay_date], key=f"replay_{replay_date}_{config}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        st.error(f"Could not complete model diagnostics: {error}")
        st.exception(error)
