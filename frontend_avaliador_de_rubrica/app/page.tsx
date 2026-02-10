"use client";

import { useState, useEffect, useCallback } from "react";

interface Item {
  index: number;
  frase: string;
  empathy: string;
  assertiveness: string;
  tone: string;
}

const EMPATHY_OPTIONS: { value: string; label: string; description: string }[] = [
  { value: "", label: "—", description: "Não classificado" },
  { value: "Baixo", label: "Baixo", description: "Ausência de reconhecimento explícito dos sentimentos ou perspectiva do interlocutor" },
  { value: "Médio", label: "Médio", description: "Reconhecimento parcial ou implícito do contexto emocional do interlocutor" },
  { value: "Alto", label: "Alto", description: "Reconhecimento explícito e validação da perspectiva ou sentimento do interlocutor" },
];

const ASSERTIVENESS_OPTIONS: { value: string; label: string; description: string }[] = [
  { value: "", label: "—", description: "Não classificado" },
  { value: "Baixo", label: "Baixo", description: "Mensagem vaga, evasiva ou excessivamente passiva" },
  { value: "Médio", label: "Médio", description: "Expressa a posição, porém com ambiguidade ou mitigação excessiva" },
  { value: "Alto", label: "Alto", description: "Expressa posição clara, objetiva e respeitosa" },
];

const TONE_OPTIONS: { value: string; label: string; description: string }[] = [
  { value: "", label: "—", description: "Não classificado" },
  { value: "Agressivo", label: "Agressivo", description: "Uso de linguagem hostil, acusatória ou depreciativa, com ausência de estratégias de mitigação ou cortesia" },
  { value: "Neutro", label: "Neutro", description: "Linguagem informativa ou descritiva, sem marcas explícitas de hostilidade ou cordialidade" },
  { value: "Amigável", label: "Amigável", description: "Linguagem cordial, respeitosa ou colaborativa, com uso de estratégias de cortesia ou suavização" },
];

function truncateFrase(text: string, maxLen = 36): string {
  const t = text.trim();
  if (t.length <= maxLen) return t;
  return t.slice(0, maxLen).trim() + "…";
}

function persistEvaluations(items: Item[]) {
  return fetch("/api/evaluations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      evaluations: items.map((i) => ({
        frase: i.frase,
        empathy: i.empathy,
        assertiveness: i.assertiveness,
        tone: i.tone,
      })),
    }),
  });
}

function RadioGroup({
  name,
  title,
  options,
  value,
  onChange,
}: {
  name: string;
  title: string;
  options: { value: string; label: string; description: string }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <fieldset className="rounded-xl border border-stone-200 dark:border-stone-600 bg-white dark:bg-stone-800/50 p-4">
      <legend className="text-sm font-semibold text-stone-700 dark:text-stone-200 px-1">
        {title}
      </legend>
      <div className="space-y-2 mt-2">
        {options.map((opt) => (
          <label
            key={opt.value}
            className={`flex gap-3 p-3 rounded-lg cursor-pointer border transition-colors ${
              value === opt.value
                ? "border-amber-500 dark:border-amber-400 bg-amber-50/50 dark:bg-amber-950/30"
                : "border-transparent hover:bg-stone-50 dark:hover:bg-stone-700/50"
            }`}
          >
            <input
              type="radio"
              name={name}
              value={opt.value}
              checked={value === opt.value}
              onChange={() => onChange(opt.value)}
              className="mt-1 h-4 w-4 border-stone-300 dark:border-stone-500 text-amber-600 dark:text-amber-400 focus:ring-amber-500"
            />
            <div className="flex-1 min-w-0">
              <span className="font-medium text-stone-800 dark:text-stone-100">
                {opt.label}
              </span>
              <p className="text-sm text-stone-600 dark:text-stone-400 mt-0.5">
                {opt.description}
              </p>
            </div>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export default function RubricEvaluator() {
  const [items, setItems] = useState<Item[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [editText, setEditText] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/data");
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setItems(data.items ?? []);
      setCurrentIndex((i) => (data.items?.length ? Math.min(i, data.items.length - 1) : 0));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const current = items[currentIndex];
  const total = items.length;
  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex < total - 1 && total > 0;

  const updateCurrent = useCallback(
    (field: keyof Item, value: string) => {
      if (!current) return;
      const next = items.map((row, i) =>
        i === currentIndex ? { ...row, [field]: value } : row
      );
      setItems(next);
      setSaving(true);
      persistEvaluations(next)
        .then((r) => {
          if (!r.ok) return r.json().then((d) => Promise.reject(new Error(d.error)));
        })
        .catch((e) => setError(e instanceof Error ? e.message : "Save failed"))
        .finally(() => setSaving(false));
    },
    [currentIndex, items, current]
  );

  const goPrev = () => {
    setCurrentIndex((i) => Math.max(0, i - 1));
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const goNext = () => {
    setCurrentIndex((i) => Math.min(total - 1, i + 1));
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const openEdit = () => {
    if (current) {
      setEditText(current.frase);
      setEditOpen(true);
    }
  };

  const saveEdit = async () => {
    if (editText === current?.frase) {
      setEditOpen(false);
      return;
    }
    setSaving(true);
    try {
      const res = await fetch("/api/edit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index: currentIndex, newFrase: editText }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      setItems((prev) =>
        prev.map((row, i) =>
          i === currentIndex ? { ...row, frase: editText } : row
        )
      );
      setEditOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Edit save failed");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-stone-50 dark:bg-stone-950">
        <p className="text-stone-500 dark:text-stone-400">Carregando…</p>
      </div>
    );
  }

  if (error && items.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-stone-50 dark:bg-stone-950 p-4">
        <p className="text-red-600 dark:text-red-400 text-center">{error}</p>
      </div>
    );
  }

  if (total === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-stone-50 dark:bg-stone-950">
        <p className="text-stone-500 dark:text-stone-400">Nenhum texto em database/input.txt</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-stone-50 dark:bg-stone-950 text-stone-900 dark:text-stone-100 flex">
      <div className="flex-1 min-w-0 flex flex-col">
        <header className="border-b border-stone-200 dark:border-stone-700 bg-white/80 dark:bg-stone-900/80 backdrop-blur sticky top-0 z-10">
          <div className="max-w-2xl mx-auto px-4 py-3">
            <div className="flex items-center justify-between">
              <h1 className="text-lg font-semibold text-stone-800 dark:text-stone-200">
                Avaliação por rubrica
              </h1>
              <span className="text-sm text-stone-500 dark:text-stone-400 tabular-nums">
                {currentIndex + 1} / {total}
              </span>
            </div>
            <p className="mt-2 text-sm text-stone-600 dark:text-stone-400">
              <strong className="text-stone-700 dark:text-stone-300">Quem é o interlocutor?</strong>{" "}
              O interlocutor é a pessoa a quem a mensagem é dirigida — ou seja, o destinatário ou receptor da fala no texto que você está avaliando. As dimensões (empatia, assertividade e tom) referem-se ao modo como o autor da mensagem se dirige a esse interlocutor.
            </p>
          </div>
        </header>

        {error && (
          <div className="max-w-2xl mx-auto px-4 py-2 bg-amber-50 dark:bg-amber-950/50 text-amber-800 dark:text-amber-200 text-sm w-full">
            {error}
          </div>
        )}

        <main className="max-w-2xl mx-auto px-4 py-6 flex-1 w-full">
        {editOpen ? (
          <div className="space-y-4">
            <label className="block text-sm font-medium text-stone-600 dark:text-stone-300">
              Editar texto
            </label>
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              className="w-full min-h-[120px] px-3 py-2 border border-stone-300 dark:border-stone-600 rounded-lg bg-white dark:bg-stone-800 text-stone-900 dark:text-stone-100 placeholder-stone-400 dark:placeholder-stone-500 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
              placeholder="Texto a avaliar"
            />
            <div className="flex gap-2">
              <button
                onClick={saveEdit}
                disabled={saving}
                className="px-4 py-2 rounded-lg bg-amber-600 dark:bg-amber-500 text-white font-medium hover:bg-amber-700 dark:hover:bg-amber-600 disabled:opacity-50"
              >
                {saving ? "Salvando…" : "Salvar (cria arquivo versionado)"}
              </button>
              <button
                onClick={() => setEditOpen(false)}
                disabled={saving}
                className="px-4 py-2 rounded-lg border border-stone-300 dark:border-stone-600 text-stone-700 dark:text-stone-300 hover:bg-stone-100 dark:hover:bg-stone-800"
              >
                Cancelar
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="mb-6">
              <p className="text-stone-800 dark:text-stone-200 text-lg leading-relaxed whitespace-pre-wrap">
                {current?.frase}
              </p>
              <button
                type="button"
                onClick={openEdit}
                className="mt-2 text-sm text-amber-600 dark:text-amber-400 hover:text-amber-700 dark:hover:text-amber-300 font-medium"
              >
                Editar este texto
              </button>
            </div>

            <div className="space-y-5">
              <RadioGroup
                name={`empathy-${currentIndex}`}
                title="Empatia"
                options={EMPATHY_OPTIONS}
                value={current?.empathy ?? ""}
                onChange={(v) => updateCurrent("empathy", v)}
              />
              <RadioGroup
                name={`assertiveness-${currentIndex}`}
                title="Assertividade"
                options={ASSERTIVENESS_OPTIONS}
                value={current?.assertiveness ?? ""}
                onChange={(v) => updateCurrent("assertiveness", v)}
              />
              <RadioGroup
                name={`tone-${currentIndex}`}
                title="Tom"
                options={TONE_OPTIONS}
                value={current?.tone ?? ""}
                onChange={(v) => updateCurrent("tone", v)}
              />
            </div>

            {saving && (
              <p className="mt-3 text-sm text-stone-500 dark:text-stone-400">Salvando…</p>
            )}
          </>
        )}
        </main>

        <footer className="border-t border-stone-200 dark:border-stone-700 bg-white/80 dark:bg-stone-900/80 mt-auto">
          <div className="max-w-2xl mx-auto px-4 py-3 flex items-center justify-between gap-4">
            <button
              type="button"
              onClick={goPrev}
              disabled={!hasPrev}
              className="px-4 py-2 rounded-lg border border-stone-300 dark:border-stone-600 text-stone-700 dark:text-stone-300 font-medium hover:bg-stone-100 dark:hover:bg-stone-800 disabled:opacity-40 disabled:pointer-events-none"
            >
              ← Anterior
            </button>
            <div className="lg:hidden overflow-x-auto flex-1 flex gap-1 px-2 min-w-0 max-w-[50vw]">
              {items.map((i) => (
                <button
                  key={i.index}
                  type="button"
                  onClick={() => { setCurrentIndex(i.index); window.scrollTo({ top: 0, behavior: "smooth" }); }}
                  className={`shrink-0 w-8 h-8 rounded text-sm font-medium transition-colors ${
                    i.index === currentIndex
                      ? "bg-amber-600 dark:bg-amber-500 text-white"
                      : "bg-stone-200 dark:bg-stone-700 text-stone-700 dark:text-stone-300"
                  }`}
                >
                  {i.index + 1}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={goNext}
              disabled={!hasNext}
              className="px-4 py-2 rounded-lg border border-stone-300 dark:border-stone-600 text-stone-700 dark:text-stone-300 font-medium hover:bg-stone-100 dark:hover:bg-stone-800 disabled:opacity-40 disabled:pointer-events-none"
            >
              Próximo →
            </button>
          </div>
        </footer>
      </div>

      <aside className="hidden lg:flex flex-col w-72 shrink-0 border-l border-stone-200 dark:border-stone-700 bg-white/60 dark:bg-stone-900/60">
        <div className="sticky top-0 py-3 px-3">
          <h2 className="text-xs font-semibold text-stone-500 dark:text-stone-400 uppercase tracking-wider px-1 mb-2">
            Ir para
          </h2>
          <nav className="overflow-y-auto max-h-[calc(100vh-5rem)] space-y-0.5">
            {items.map((item, i) => (
              <button
                key={i}
                type="button"
                onClick={() => { setCurrentIndex(i); window.scrollTo({ top: 0, behavior: "smooth" }); }}
                className={`w-full text-left px-2 py-1.5 rounded-md text-sm transition-colors ${
                  i === currentIndex
                    ? "bg-amber-100 dark:bg-amber-900/40 text-amber-900 dark:text-amber-100 font-medium"
                    : "text-stone-600 dark:text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-800 hover:text-stone-900 dark:hover:text-stone-100"
                }`}
                title={item.frase}
              >
                <span className="tabular-nums text-stone-400 dark:text-stone-500 mr-1.5">
                  {i + 1}.
                </span>
                {truncateFrase(item.frase)}
              </button>
            ))}
          </nav>
        </div>
      </aside>
    </div>
  );
}
