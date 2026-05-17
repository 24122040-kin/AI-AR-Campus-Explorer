"""
scripts/cli.py — Command line interface for LocalNavBot
Usage:
  python cli.py serve          # Start the API server
  python cli.py index [folder] # Index images from a folder
  python cli.py add-location   # Interactive wizard to add a location
  python cli.py demo           # Run a demo query
  python cli.py status         # Show system status
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from loguru import logger

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = typer.Typer(help="LocalNavBot — Local Navigation Assistant", rich_markup_mode="rich")
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# serve
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload"),
):
    """Start the LocalNavBot API server."""
    import uvicorn
    console.print(Panel.fit(
        "[bold green]🚀 LocalNavBot starting[/bold green]\n"
        f"[cyan]http://{host}:{port}[/cyan]",
        title="Server"
    ))
    uvicorn.run(
        "web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ─────────────────────────────────────────────────────────────────────────────
# index
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def index(
    folder: Path = typer.Argument(Path("data/images"), help="Folder containing GPS-tagged images"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Force rebuild VLAD vocabulary"),
):
    """Index all images in a folder into the VPR database."""
    async def _run():
        from config.settings import settings
        settings.setup_dirs()

        from core.database import db
        await db.init()

        from core.vpr_engine import VPREngine, ImageMeta

        if not folder.exists():
            console.print(f"[red]Folder not found: {folder}[/red]")
            raise typer.Exit(1)

        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        images = [p for p in folder.rglob("*") if p.suffix.lower() in image_exts]

        if not images:
            console.print(f"[yellow]No images found in {folder}[/yellow]")
            raise typer.Exit(0)

        console.print(f"Found [bold]{len(images)}[/bold] images")

        vpr = VPREngine()

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
            task = prog.add_task("Processing images…", total=len(images))

            # Build locations from GPS EXIF + create DB entries
            metas: list[ImageMeta] = []
            for img_path in images:
                gps = VPREngine.gps_from_exif(img_path)
                if gps is None:
                    prog.console.print(f"  [yellow]Skip (no GPS):[/yellow] {img_path.name}")
                    prog.advance(task)
                    continue

                lat, lon = gps
                # Check if location already exists nearby
                nearby = await db.nearby_locations(lat, lon, radius_deg=0.00005)  # ~5m
                if nearby:
                    loc_id = nearby[0]["id"]
                    loc_name = nearby[0]["name"]
                else:
                    loc_name = img_path.stem.replace("_", " ").replace("-", " ").title()
                    loc_id = await db.add_location(name=loc_name, lat=lat, lon=lon)

                img_id = await db.add_image(
                    location_id=loc_id,
                    filename=img_path.name,
                    filepath=str(img_path),
                )
                metas.append(ImageMeta(
                    image_id=img_id, location_id=loc_id,
                    location_name=loc_name, lat=lat, lon=lon,
                    filepath=str(img_path),
                ))
                prog.advance(task)

        if not metas:
            console.print("[red]No GPS-tagged images found. Ensure EXIF GPS data is present.[/red]")
            raise typer.Exit(1)

        console.print(f"[green]Indexed metadata for {len(metas)} images[/green]")

        # Build / rebuild VPR index
        if rebuild or not vpr.aggregator._fitted:
            console.print("Building VLAD vocabulary and FAISS index (may take a few minutes on GPU)…")
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
                t = prog.add_task("Encoding…")
                vpr.index_all_images(metas)
                prog.stop_task(t)
        else:
            console.print("[cyan]VLAD vocab found. Adding new images to existing index…[/cyan]")
            for meta in metas:
                if meta.faiss_idx == -1:
                    try:
                        faiss_id = vpr.index_image(Path(meta.filepath), meta)
                        await db.update_faiss_id(meta.image_id, faiss_id)
                    except Exception as e:
                        console.print(f"  [yellow]Skip {meta.filepath}: {e}[/yellow]")
            vpr._index.save()

        console.print(Panel.fit(
            f"[bold green]✓ VPR index built[/bold green]\n"
            f"Total images indexed: {vpr._ensure_index().size}\n"
            f"Locations in DB: {len(metas)}",
            title="Done"
        ))

    asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# add-location (interactive wizard)
# ─────────────────────────────────────────────────────────────────────────────

@app.command("add-location")
def add_location_wizard():
    """Interactive wizard to add a location with photos."""
    async def _run():
        from config.settings import settings
        settings.setup_dirs()
        from core.database import db
        await db.init()

        console.print(Panel.fit("[bold]Add a new location[/bold]", title="LocalNavBot"))

        name        = typer.prompt("Location name")
        lat         = float(typer.prompt("Latitude (e.g. 10.9085)"))
        lon         = float(typer.prompt("Longitude (e.g. 106.7600)"))
        description = typer.prompt("Description (optional)", default="")
        category    = typer.prompt("Category (general/cafe/landmark/alley/shortcut)", default="general")
        importance  = int(typer.prompt("Importance 1-5", default="1"))
        img_paths_s = typer.prompt("Image paths (comma-separated, optional)", default="")

        loc_id = await db.add_location(
            name=name, lat=lat, lon=lon,
            description=description, category=category,
            importance=importance,
        )

        if img_paths_s.strip():
            from core.vpr_engine import VPREngine, ImageMeta
            vpr = VPREngine()
            for p_str in img_paths_s.split(","):
                p = Path(p_str.strip())
                if not p.exists():
                    console.print(f"  [yellow]File not found: {p}[/yellow]")
                    continue
                img_id = await db.add_image(
                    location_id=loc_id,
                    filename=p.name,
                    filepath=str(p),
                )
                if vpr.aggregator._fitted:
                    meta = ImageMeta(
                        image_id=img_id, location_id=loc_id,
                        location_name=name, lat=lat, lon=lon,
                        filepath=str(p),
                    )
                    faiss_id = vpr.index_image(p, meta)
                    await db.update_faiss_id(img_id, faiss_id)
                    vpr._index.save()

        console.print(f"[green]✓ Location '{name}' added (id={loc_id})[/green]")

    asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# demo
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def demo(
    query: str = typer.Argument("Đường nào đi từ Dĩ An đến chợ Bình Dương ít tắc nhất?"),
    lat: Optional[float] = typer.Option(None, "--lat"),
    lon: Optional[float] = typer.Option(None, "--lon"),
):
    """Run a demo query through the bot."""
    async def _run():
        from config.settings import settings
        settings.setup_dirs()
        from core.database import db
        await db.init()
        from core.vpr_engine import VPREngine
        from routing.router import NavRouter
        from bot.nav_bot import NavBot

        console.print(f"[cyan]Query:[/cyan] {query}")

        router = NavRouter()
        await router.init()
        vpr = VPREngine()
        bot = NavBot(router, vpr)

        with Progress(SpinnerColumn(), TextColumn("Thinking…"), console=console) as p:
            t = p.add_task("")
            response = await bot.ask(query, user_lat=lat, user_lon=lon)
            p.stop_task(t)

        console.print(Panel(response, title="Bot response", expand=False))

    asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# status
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def status():
    """Show system status."""
    async def _run():
        from config.settings import settings
        settings.setup_dirs()
        from core.database import db
        await db.init()

        loc_count  = (await db.fetchone("SELECT COUNT(*) AS n FROM locations") or {}).get("n", 0)
        img_count  = (await db.fetchone("SELECT COUNT(*) AS n FROM images") or {}).get("n", 0)
        poi_count  = (await db.fetchone("SELECT COUNT(*) AS n FROM pois") or {}).get("n", 0)
        edge_count = (await db.fetchone("SELECT COUNT(*) AS n FROM custom_edges") or {}).get("n", 0)

        t = Table(title="LocalNavBot Status", show_header=True)
        t.add_column("Component")
        t.add_column("Status")
        t.add_column("Details")

        try:
            import torch

            gpu_status = "✓ Available" if torch.cuda.is_available() else "✗ CPU only"
            gpu_details = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        except Exception as e:
            gpu_status = "✗ Torch unavailable"
            gpu_details = str(e)
        t.add_row("GPU", gpu_status, gpu_details)

        faiss_idx = settings.faiss_index_path
        t.add_row("FAISS index", "✓ Exists" if faiss_idx.exists() else "✗ Not built",
                  str(faiss_idx))

        db_path = settings.db_path
        t.add_row("Database", "✓ Ready" if db_path.exists() else "✗ Missing",
                  f"{loc_count} locations, {img_count} images, {poi_count} POIs, {edge_count} edges")

        osm_cache = settings.osm_cache_dir
        osm_files = list(osm_cache.glob("*.graphml")) if osm_cache.exists() else []
        t.add_row("OSM cache", "✓ Cached" if osm_files else "✗ Not downloaded",
                  str(osm_files[0].name) if osm_files else "Run 'serve' to auto-download")

        t.add_row("LLM", f"{settings.llm_provider}", f"{settings.llm_model}")
        t.add_row("VPR model", settings.vpr_model, f"layer {settings.vpr_layer}, {settings.vpr_facet}")

        console.print(t)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
