import { NextRequest, NextResponse } from "next/server";

/**
 * Proxies chart PNGs from the Python server (/charts/...) for use in <img> tags.
 */
export async function GET(
  _request: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  const base = process.env.AI_MODEL_API_BASE_URL?.replace(/\/$/, "");
  if (!base) {
    return new NextResponse("AI layer not configured", { status: 503 });
  }

  const { path } = await context.params;
  if (!path?.length) {
    return new NextResponse("Missing chart path", { status: 400 });
  }

  const rel = path.map((p) => encodeURIComponent(p)).join("/");
  const url = `${base}/charts/${rel}`;

  try {
    const res = await fetch(url);
    if (!res.ok) {
      return new NextResponse("Chart not found", { status: res.status });
    }
    const buf = await res.arrayBuffer();
    const ct = res.headers.get("content-type") ?? "image/png";
    return new NextResponse(buf, {
      headers: {
        "Content-Type": ct,
        "Cache-Control": "public, max-age=300",
      },
    });
  } catch (e) {
    console.error("chart proxy:", e);
    return new NextResponse("Failed to fetch chart", { status: 502 });
  }
}
