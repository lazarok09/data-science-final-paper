import { NextRequest, NextResponse } from "next/server";
import { writeFileSync, readFileSync, existsSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

function getDbDir(): string {
  const candidates = [
    join(process.cwd(), "database"),
    join(__dirname, "..", "..", "..", "database"),
    join(__dirname, "..", "..", "..", "..", "..", "database"),
  ];
  for (const dir of candidates) {
    if (existsSync(join(dir, "input.txt"))) return dir;
  }
  return candidates[0];
}

const DB_DIR = getDbDir();
const CSV_PATH = join(DB_DIR, "output.csv");

function escapeCsv(s: string): string {
  const t = String(s ?? "");
  if (t.includes('"') || t.includes(",") || t.includes("\n")) {
    return `"${t.replace(/"/g, '""')}"`;
  }
  return t;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { evaluations } = body as {
      evaluations: { frase: string; empathy?: string; assertiveness?: string; tone?: string }[];
    };
    if (!Array.isArray(evaluations)) {
      return NextResponse.json(
        { error: "evaluations array required" },
        { status: 400 }
      );
    }
    if (!existsSync(DB_DIR)) {
      mkdirSync(DB_DIR, { recursive: true });
    }
    const header = "frase,empathy,assertiveness,tone";
    const csvLines = [
      header,
      ...evaluations.map(
        (r) =>
          `${escapeCsv(r.frase)},${escapeCsv(r.empathy ?? "")},${escapeCsv(r.assertiveness ?? "")},${escapeCsv(r.tone ?? "")}`
      ),
    ];
    const csvContent = csvLines.join("\n") + "\n";
    writeFileSync(CSV_PATH, csvContent, "utf-8");
    return NextResponse.json({ ok: true });
  } catch (e) {
    console.error(e);
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Failed to save" },
      { status: 500 }
    );
  }
}
