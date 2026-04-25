from __future__ import annotations

from datetime import date, datetime, timezone
from html import escape
from pathlib import Path

from paper_alpha_agent.models.paper import FullPaperSummary, SummarizedPaper, TopicSummaryStageResult


class TopicSummaryEmailHtmlWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        result: TopicSummaryStageResult,
        start_date: str | None = None,
        end_date: str | None = None,
        output_path: Path | None = None,
    ) -> Path:
        created_at = datetime.now(timezone.utc)
        path = output_path or self.output_dir / (
            f"topic_summary_email_{self._filename_window_suffix(start_date=start_date, end_date=end_date)}_"
            f"{created_at.strftime('%Y%m%d_%H%M%S')}.html"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self._render(result=result, created_at=created_at, start_date=start_date, end_date=end_date),
            encoding="utf-8",
        )
        return path

    def _render(
        self,
        result: TopicSummaryStageResult,
        created_at: datetime,
        start_date: str | None,
        end_date: str | None,
    ) -> str:
        selected_ids = {paper.paper_id for paper in result.full_paper_summaries}
        table_rows_html = "\n".join(
            self._render_table_row(paper=paper, selected_for_full_text=paper.paper_id in selected_ids)
            for paper in result.filtered_papers
        )
        summary_cards_html = "\n".join(
            self._render_summary_card(index=index, paper=paper)
            for index, paper in enumerate(result.full_paper_summaries, start=1)
        )
        date_window = f"{self._display(start_date)} to {self._display(end_date)}"

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Paper Alpha Agent Topic Summary</title>
  <style>
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 16px;
      background: #f4f6f8;
      color: #1f2937;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      line-height: 1.5;
    }}
    .container {{
      max-width: 980px;
      margin: 0 auto;
    }}
    .hero {{
      background: linear-gradient(135deg, #ecfeff, #eff6ff);
      border: 1px solid #dbeafe;
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      line-height: 1.3;
      color: #0f172a;
    }}
    .meta {{
      margin: 0;
      font-size: 14px;
      color: #334155;
    }}
    .section {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 14px;
      padding: 14px;
      margin-bottom: 14px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 20px;
      color: #111827;
    }}
    .hint {{
      margin: 0 0 12px;
      color: #4b5563;
      font-size: 14px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      -webkit-overflow-scrolling: touch;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 980px;
      font-size: 13px;
      background: #ffffff;
    }}
    th, td {{
      border-bottom: 1px solid #f1f5f9;
      text-align: left;
      padding: 10px 8px;
      vertical-align: top;
    }}
    th {{
      background: #f8fafc;
      color: #334155;
      font-size: 12px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      white-space: nowrap;
    }}
    td.num {{
      text-align: right;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}
    td.topic {{
      white-space: nowrap;
    }}
    td.link {{
      white-space: nowrap;
    }}
    td.link a {{
      color: #1d4ed8;
      text-decoration: none;
    }}
    td.link a:hover {{
      text-decoration: underline;
    }}
    td.title {{
      min-width: 320px;
    }}
    .paper-card {{
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 14px;
      background: #ffffff;
      margin-bottom: 12px;
    }}
    .paper-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }}
    .tag {{
      background: #e0f2fe;
      color: #0369a1;
      border: 1px solid #bae6fd;
      border-radius: 999px;
      padding: 3px 10px;
      font-size: 12px;
      font-weight: 600;
    }}
    .paper-link {{
      font-size: 13px;
      color: #1d4ed8;
      text-decoration: none;
      white-space: nowrap;
    }}
    .paper-link:hover {{
      text-decoration: underline;
    }}
    .paper-title {{
      margin: 0 0 10px;
      font-size: 18px;
      line-height: 1.35;
      color: #111827;
    }}
    .kv-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}
    .kv-item {{
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 8px;
    }}
    .kv-label {{
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: #64748b;
      margin-bottom: 4px;
    }}
    .kv-value {{
      font-size: 14px;
      color: #0f172a;
      word-break: break-word;
    }}
    .subsection {{
      margin-bottom: 10px;
    }}
    .subsection h3 {{
      margin: 0 0 6px;
      font-size: 14px;
      color: #0f172a;
    }}
    .subsection p {{
      margin: 0 0 6px;
      font-size: 14px;
      color: #1f2937;
    }}
    .subsection ul {{
      margin: 0;
      padding-left: 18px;
      color: #1f2937;
      font-size: 14px;
    }}
    .empty {{
      margin: 0;
      color: #6b7280;
      font-style: italic;
    }}
    @media (max-width: 700px) {{
      body {{
        padding: 10px;
      }}
      .hero, .section, .paper-card {{
        padding: 12px;
      }}
      h1 {{
        font-size: 20px;
      }}
      h2 {{
        font-size: 18px;
      }}
      .paper-title {{
        font-size: 16px;
      }}
      .kv-grid {{
        grid-template-columns: 1fr;
      }}
      table {{
        font-size: 12px;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <header class="hero">
      <h1>Paper Alpha Agent Topic Summary</h1>
      <p class="meta"><strong>Created:</strong> {escape(created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))}</p>
      <p class="meta"><strong>Date window:</strong> {escape(date_window)}</p>
      <p class="meta"><strong>Topics:</strong> {escape(", ".join(result.topics))}</p>
      <p class="meta"><strong>Filtered candidates:</strong> {len(result.filtered_papers)} | <strong>Full-paper summaries:</strong> {len(result.full_paper_summaries)}</p>
    </header>

    <section class="section">
      <h2>Final Filtered Candidates</h2>
      <p class="hint">Directly relevant papers with implementable alpha label in <code>yes</code> or <code>likely</code>.</p>
      {self._render_candidates_table(table_rows_html)}
    </section>

    <section class="section">
      <h2>Full-Paper Summaries</h2>
      <p class="hint">Each summary card includes structured fields and a direct link to the source paper.</p>
      {summary_cards_html or '<p class="empty">No full-paper summaries were generated.</p>'}
    </section>
  </div>
</body>
</html>
"""

    def _render_candidates_table(self, table_rows_html: str) -> str:
        if not table_rows_html:
            return '<p class="empty">No filtered candidates available.</p>'
        return f"""
<div class="table-wrap">
  <table role="presentation">
    <thead>
      <tr>
        <th>Selected</th>
        <th>Alpha</th>
        <th>Global Rank</th>
        <th>Topic Rank</th>
        <th>Topic</th>
        <th>Link</th>
        <th>Title</th>
      </tr>
    </thead>
    <tbody>
      {table_rows_html}
    </tbody>
  </table>
</div>
"""

    def _render_table_row(self, paper: SummarizedPaper, selected_for_full_text: bool) -> str:
        link = self._paper_link(paper)
        link_html = (
            f'<a href="{escape(link, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(link)}</a>'
            if link
            else "N/A"
        )
        return f"""
<tr>
  <td>{'yes' if selected_for_full_text else 'no'}</td>
  <td>{escape(self._display(paper.implementable_alpha_label))}</td>
  <td class="num">{escape(self._display(paper.global_rank))}</td>
  <td class="num">{escape(self._display(paper.summary_rank))}</td>
  <td class="topic">{escape(self._display(paper.query_topic))}</td>
  <td class="link">{link_html}</td>
  <td class="title">{escape(self._display(paper.title))}</td>
</tr>
"""

    def _render_summary_card(self, index: int, paper: FullPaperSummary) -> str:
        link = self._paper_link(paper)
        link_html = (
            f'<a class="paper-link" href="{escape(link, quote=True)}" target="_blank" rel="noopener noreferrer">Open paper on arXiv</a>'
            if link
            else '<span class="paper-link">arXiv link unavailable</span>'
        )

        return f"""
<article class="paper-card">
  <div class="paper-head">
    <span class="tag">PaperSummary #{index}</span>
    {link_html}
  </div>
  <h3 class="paper-title">{escape(self._display(paper.title))}</h3>
  <div class="kv-grid">
    <div class="kv-item"><span class="kv-label">Global Rank</span><span class="kv-value">{escape(self._display(paper.global_rank))}</span></div>
    <div class="kv-item"><span class="kv-label">Topic Rank</span><span class="kv-value">{escape(self._display(paper.summary_rank))}</span></div>
    <div class="kv-item"><span class="kv-label">Topic</span><span class="kv-value">{escape(self._display(paper.query_topic))}</span></div>
    <div class="kv-item"><span class="kv-label">Alpha</span><span class="kv-value">{escape(self._display(paper.implementable_alpha_label))}</span></div>
    <div class="kv-item"><span class="kv-label">Complexity</span><span class="kv-value">{escape(self._display(paper.implementation_complexity))}</span></div>
    <div class="kv-item"><span class="kv-label">Quality</span><span class="kv-value">{escape(self._display(paper.strategy_quality))}</span></div>
    <div class="kv-item"><span class="kv-label">Sharpe</span><span class="kv-value">{escape(self._display(paper.sharpe_ratio))}</span></div>
    <div class="kv-item"><span class="kv-label">Sharpe Note</span><span class="kv-value">{escape(self._display(paper.sharpe_ratio_context))}</span></div>
  </div>
  {self._render_optional_text_section("Alpha Thesis", paper.alpha_thesis)}
  {self._render_summary_section("Summary", paper.full_text_summary)}
  {self._render_optional_list_section("Evidence", paper.evidence)}
  {self._render_optional_list_section("Implementation Requirements", paper.implementation_requirements)}
  {self._render_optional_list_section("Key Risks", paper.key_risks)}
</article>
"""

    def _render_optional_text_section(self, title: str, text: str | None) -> str:
        if not text:
            return ""
        return f"""
<div class="subsection">
  <h3>{escape(title)}</h3>
  <p>{escape(text)}</p>
</div>
"""

    def _render_summary_section(self, title: str, text: str | None) -> str:
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        if not lines:
            lines = ["None"]
        paragraphs = "\n".join(f"<p>{escape(line)}</p>" for line in lines)
        return f"""
<div class="subsection">
  <h3>{escape(title)}</h3>
  {paragraphs}
</div>
"""

    def _render_optional_list_section(self, title: str, values: list[str]) -> str:
        if not values:
            return ""
        items = "\n".join(f"<li>{escape(value)}</li>" for value in values if value)
        if not items:
            return ""
        return f"""
<div class="subsection">
  <h3>{escape(title)}</h3>
  <ul>
    {items}
  </ul>
</div>
"""

    @staticmethod
    def _paper_link(paper: SummarizedPaper | FullPaperSummary) -> str:
        return str(paper.entry_url or paper.pdf_url or "")

    @staticmethod
    def _display(value: object) -> str:
        return "None" if value is None else str(value)

    @classmethod
    def _filename_window_suffix(cls, start_date: str | None, end_date: str | None) -> str:
        end_fragment = cls._safe_filename_fragment(end_date or "unknown")
        span_fragment = "unknown"
        parsed_start = cls._parse_date(start_date)
        parsed_end = cls._parse_date(end_date)
        if parsed_start is not None and parsed_end is not None:
            span_days = (parsed_end - parsed_start).days + 1
            if span_days > 0:
                span_fragment = f"{span_days}d"
        return f"{end_fragment}_{span_fragment}"

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        if not value:
            return None
        candidate = value.strip()
        if len(candidate) >= 10:
            try:
                return date.fromisoformat(candidate[:10])
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).date()
        except ValueError:
            return None

    @staticmethod
    def _safe_filename_fragment(value: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value).strip("-") or "unknown"
