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
const INPUT_PATH = join(DB_DIR, "input.txt");

function escapeCsv(s: string): string {
  const t = String(s ?? "");
  if (t.includes('"') || t.includes(",") || t.includes("\n")) {
    return `"${t.replace(/"/g, '""')}"`;
  }
  return t;
}

function parseCsvLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      inQuotes = !inQuotes;
    } else if (c === "," && !inQuotes) {
      result.push(current.trim().replace(/^"|"$/g, "").replace(/""/g, '"'));
      current = "";
    } else {
      current += c;
    }
  }
  result.push(current.trim().replace(/^"|"$/g, "").replace(/""/g, '"'));
  return result;
}

function versionedFilename(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  const y = d.getFullYear();
  const m = pad(d.getMonth() + 1);
  const day = pad(d.getDate());
  const h = pad(d.getHours());
  const min = pad(d.getMinutes());
  const s = pad(d.getSeconds());
  return `output_${y}_${m}_${day}_${h}_${min}_${s}.csv`;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { index, newFrase } = body as { index: number; newFrase: string };
    if (typeof index !== "number" || index < 0 || newFrase === undefined) {
      return NextResponse.json(
        { error: "index (number) and newFrase required" },
        { status: 400 }
      );
    }
    if (!existsSync(DB_DIR)) {
      mkdirSync(DB_DIR, { recursive: true });
    }
    let dataLines: string[];
    const header = "frase,empathy,assertiveness,tone";
    if (!existsSync(CSV_PATH)) {
      if (!existsSync(INPUT_PATH)) {
        return NextResponse.json(
          { error: "database/input.txt not found" },
          { status: 404 }
        );
      }
      const inputContent = readFileSync(INPUT_PATH, "utf-8");
      const inputLines = inputContent
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean);
      dataLines = inputLines.map((frase) =>
        [escapeCsv(frase), "", "", ""].join(",")
      );
    } else {
      const csvContent = readFileSync(CSV_PATH, "utf-8");
      const lines = csvContent.split("\n").filter(Boolean);
      dataLines = lines.slice(1);
    }
    if (index >= dataLines.length) {
      return NextResponse.json(
        { error: "index out of range" },
        { status: 400 }
      );
    }
    const row = parseCsvLine(dataLines[index]);
    const frase = row[0] ?? "";
    const empathy = row[1] ?? "";
    const assertiveness = row[2] ?? "";
    const tone = row[3] ?? "";
    const newRow = [
      escapeCsv(newFrase),
      escapeCsv(empathy),
      escapeCsv(assertiveness),
      escapeCsv(tone),
    ].join(",");
    dataLines[index] = newRow;
    const newCsv = [header, ...dataLines].join("\n") + "\n";
    writeFileSync(CSV_PATH, newCsv, "utf-8");

    const versionedPath = join(DB_DIR, versionedFilename());
    writeFileSync(versionedPath, newCsv, "utf-8");

    return NextResponse.json({
      ok: true,
      versioned: versionedPath.split("/").pop(),
    });
  } catch (e) {
    console.error(e);
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Failed to save" },
      { status: 500 }
    );
  }
}
