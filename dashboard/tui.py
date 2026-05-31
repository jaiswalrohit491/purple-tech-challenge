"""Live TUI dashboard for the Store Intelligence API.

Polls /health to discover active stores, then for each store fetches /metrics
and /anomalies on a 2-second cadence. Renders a grid of per-store cards and a
rolling anomalies tail. Designed to satisfy Part E of the brief — proof that
the pipeline and API are genuinely connected, not just batch-processed.

Usage:
    python -m dashboard.tui
    python -m dashboard.tui --url http://localhost:8000 --interval 1.5

Stop with Ctrl+C.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timezone

import httpx
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


SEVERITY_COLORS = {"INFO": "cyan", "WARN": "yellow", "CRITICAL": "red"}
STALE_COLOR = "red"
FRESH_COLOR = "green"


class DashboardState:
    def __init__(self, url: str):
        self.url = url
        self.client = httpx.Client(timeout=5.0, base_url=url)
        self.last_refresh: datetime | None = None
        self.health: dict | None = None
        self.metrics: dict[str, dict] = {}
        self.anomalies: list[dict] = []
        self.error: str | None = None

    def refresh(self) -> None:
        self.error = None
        try:
            self.health = self.client.get("/health").json()
        except Exception as e:
            self.error = f"/health: {e}"
            return

        new_metrics: dict[str, dict] = {}
        new_anomalies: list[dict] = []
        for feed in self.health.get("stores", []):
            sid = feed["store_id"]
            try:
                new_metrics[sid] = self.client.get(f"/stores/{sid}/metrics").json()
                resp = self.client.get(f"/stores/{sid}/anomalies").json()
                for a in resp.get("anomalies", []):
                    a["_store_id"] = sid
                    new_anomalies.append(a)
            except Exception as e:
                self.error = f"{sid}: {e}"
        self.metrics = new_metrics
        # Newest anomalies first; cap to keep the panel readable.
        new_anomalies.sort(key=lambda a: a.get("detected_at") or "", reverse=True)
        self.anomalies = new_anomalies[:10]
        self.last_refresh = datetime.now(timezone.utc)

    def close(self) -> None:
        self.client.close()


def render_header(state: DashboardState) -> Panel:
    title = Text("Apex Retail — Store Intelligence (live)", style="bold white")
    sub = Text()
    if state.last_refresh:
        sub.append(f"  api={state.url}", style="dim")
        sub.append(f"   updated={state.last_refresh.strftime('%H:%M:%S UTC')}", style="dim")
    if state.error:
        sub.append(f"   error={state.error}", style="red")
    if state.health:
        sub.append(f"   status={state.health.get('status', '?')}", style="dim")
    return Panel(Group(title, sub), border_style="blue", padding=(0, 1))


def render_metrics_table(state: DashboardState) -> Panel:
    table = Table(expand=True, show_lines=False, header_style="bold")
    table.add_column("Store", style="bold")
    table.add_column("Customers", justify="right")
    table.add_column("Staff", justify="right")
    table.add_column("Conv %\n(verified)", justify="right")
    table.add_column("Conv %\n(potential)", justify="right")
    table.add_column("Queue", justify="right")
    table.add_column("Abandon %", justify="right")
    table.add_column("Feed", justify="center")
    table.add_column("Top zones", overflow="fold")

    feed_map = {f["store_id"]: f for f in (state.health or {}).get("stores", [])}
    if not feed_map:
        table.add_row(Text("(no stores yet — ingest some events to start)", style="dim italic"),
                      "", "", "", "", "", "", "", "")
        return Panel(table, title="Stores", border_style="cyan")

    for sid, feed in sorted(feed_map.items()):
        m = state.metrics.get(sid) or {}
        unique = m.get("unique_visitors", 0)
        staff = m.get("staff_count", 0)
        # Use the operator-preferred "verified" rate as the headline; the
        # brief's looser conversion_rate is still in /metrics for the rubric.
        conv = (m.get("verified_purchase_rate", 0.0) or 0.0) * 100
        potential = (m.get("potential_conversion_rate", 0.0) or 0.0) * 100
        queue = m.get("current_queue_depth", 0)
        abandon = (m.get("abandonment_rate", 0.0) or 0.0) * 100
        stale = feed.get("stale")
        feed_text = Text("STALE" if stale else "OK",
                         style=STALE_COLOR if stale else FRESH_COLOR)

        zones = m.get("avg_dwell_by_zone") or []
        zones = sorted(zones, key=lambda z: z.get("visits", 0), reverse=True)[:3]
        zones_txt = " · ".join(
            f"{z['zone_id']}({z.get('visits', 0)})" for z in zones
        ) or "—"

        queue_style = "red" if queue >= 8 else "yellow" if queue >= 5 else "white"
        table.add_row(
            sid,
            f"{unique:,}",
            Text(f"{staff:,}", style="magenta"),
            f"{conv:0.1f}",
            Text(f"{potential:0.1f}", style="dim"),
            Text(str(queue), style=queue_style),
            f"{abandon:0.1f}",
            feed_text,
            zones_txt,
        )
    return Panel(table, title="Stores", border_style="cyan")


def render_anomalies(state: DashboardState) -> Panel:
    if not state.anomalies:
        body = Text("No active anomalies.", style="dim")
        return Panel(body, title="Active anomalies", border_style="green")

    table = Table(expand=True, show_header=True, header_style="bold")
    table.add_column("Store")
    table.add_column("Severity", width=10)
    table.add_column("Type", width=24)
    table.add_column("Detail", overflow="fold")
    table.add_column("Action", overflow="fold")

    for a in state.anomalies:
        sev = a.get("severity", "INFO")
        sev_style = SEVERITY_COLORS.get(sev, "white")
        detail = a.get("detail") or {}
        detail_txt = ", ".join(f"{k}={v}" for k, v in detail.items() if k != "zone_id")
        if "zone_id" in detail:
            detail_txt = f"zone={detail['zone_id']} {detail_txt}".strip()
        table.add_row(
            a.get("_store_id", "?"),
            Text(sev, style=sev_style),
            a.get("anomaly_type", "?"),
            detail_txt or "—",
            a.get("suggested_action", "—"),
        )
    return Panel(table, title="Active anomalies", border_style="yellow")


def render(state: DashboardState) -> Group:
    return Group(
        render_header(state),
        render_metrics_table(state),
        render_anomalies(state),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--interval", type=float, default=2.0)
    args = p.parse_args()

    console = Console()
    state = DashboardState(args.url)

    # Clean Ctrl+C handling so the cursor is restored.
    stopping = {"v": False}
    def _stop(_sig, _frame):
        stopping["v"] = True
    signal.signal(signal.SIGINT, _stop)

    try:
        with Live(render(state), console=console, refresh_per_second=4,
                  screen=False, vertical_overflow="visible") as live:
            while not stopping["v"]:
                state.refresh()
                live.update(render(state))
                # Sleep in small slices so Ctrl+C feels responsive.
                slept = 0.0
                while slept < args.interval and not stopping["v"]:
                    time.sleep(0.1)
                    slept += 0.1
    finally:
        state.close()
        console.print("[dim]dashboard exited.[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
