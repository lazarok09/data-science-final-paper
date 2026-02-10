import { NextResponse } from "next/server";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

const DB_DIR = join(process.cwd(), "database");
const INPUT_PATH = join(DB_DIR, "input.txt");
const CSV_PATH = join(DB_DIR, "output.csv");

export interface EvaluationRow {
  index: number;
  frase: string;
  empathy: string;
  assertiveness: string;
  tone: string;
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
      result.push(current.trim());
      current = "";
    } else {
      current += c;
    }
  }
  result.push(current.trim());
  return result;
}

export async function GET() {
  try {
    if (!existsSync(INPUT_PATH)) {
      return NextResponse.json(
        { error: "database/input.txt not found" },
        { status: 404 }
      );
    }
    const inputContent = readFileSync(INPUT_PATH, "utf-8");
    const lines = inputContent
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);

    let rows: EvaluationRow[] = lines.map((frase, index) => ({
      index,
      frase,
      empathy: "",
      assertiveness: "",
      tone: "",
    }));

    if (existsSync(CSV_PATH)) {
      const csvContent = readFileSync(CSV_PATH, "utf-8");
      const csvLines = csvContent.split("\n").filter(Boolean);
      const header = csvLines[0]?.toLowerCase() ?? "";
      const dataLines = csvLines.slice(1);
      const hasFrase = header.includes("frase");
      const hasEmpathy = header.includes("empathy") || header.includes("empatia");
      const hasAssertiveness =
        header.includes("assertiveness") || header.includes("assertividade");
      const hasTone = header.includes("tone") || header.includes("tom");

      dataLines.forEach((line, i) => {
        if (i >= rows.length) return;
        const cells = parseCsvLine(line);
        const fraseVal = hasFrase ? (cells[0] ?? "").replace(/^"|"$/g, "") : "";
        const empathyVal = hasEmpathy ? (cells[1] ?? "").replace(/^"|"$/g, "") : "";
        const assertivenessVal = hasAssertiveness
          ? (cells[2] ?? "").replace(/^"|"$/g, "")
          : "";
        const toneVal = hasTone ? (cells[3] ?? "").replace(/^"|"$/g, "") : "";
        rows[i] = {
          index: i,
          frase: fraseVal || rows[i].frase,
          empathy: empathyVal,
          assertiveness: assertivenessVal,
          tone: toneVal,
        };
      });
    }

    return NextResponse.json({ items: rows, total: rows.length });
  } catch (e) {
    console.error(e);
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Failed to load data" },
      { status: 500 }
    );
  }
}
