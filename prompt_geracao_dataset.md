# Documentação dos prompts utilizados para geração do dataset

Este arquivo registra os prompts utilizados para geração do conjunto de dados sintéticos (frases em dialeto recifense) conforme descrito nos documentos de insights e na metodologia do trabalho. O objetivo é permitir **replicabilidade**: qualquer pesquisador pode reproduzir a geração usando o mesmo prompt e um modelo de linguagem (LLM).

**Contexto:** Em razão das mudanças nas políticas de comentários em redes sociais e do uso desse conteúdo por empresas, optou-se por um dataset sintético controlado em vez de coleta por crawling. O prompt foi evoluindo: um primeiro mais curto, um segundo detalhado (com regras, tons, gírias e exemplos) e uma versão consolidada em uma linha para uso rápido.

---

## 1. Primeiro prompt (versão inicial)

Utilizado na etapa inicial de geração. Foco em quantidade, tamanho e variedade de tons.

```
Exemplifique frases regionais recifenses entre no mínimo 30 e na média até 255 caracteres com uma variação entre 255 e 300 caso ultrapasse e que sejam de tons variados entre reclamação, agressividade, xingamento, elogios, super elogios, elogio moderado, reclamação moderada, neutros altos e baixos.
```

---

## 2. Segundo prompt (estruturado e detalhado)

Prompt expandido com objetivo, regras de conteúdo, tons obrigatórios, gírias recifenses, estrutura e exemplos de frases desejadas.

### Objetivo

Gerar frases naturais no dialeto recifense, com grande variedade de gírias e nuances de emoção, ideais para análise de comportamento, CNV ou construção de dataset textual.

### Regras de conteúdo

**1) Quantidade e formato**

- Gerar 120 frases inicialmente (em CSV ou texto com linhas separadas).
- Cada linha: uma frase, pronta para copiar e colar.
- Intervalo de caracteres: 30–255 (até 300 em casos que exigem profundidade).

**2) Tons/estilos obrigatórios**

Distribuir as frases entre:

- **Reclamação:** explícita; moderada.
- **Agressividade/xingamento:** incluindo gírias e insultos recifenses.
- **Elogio:** moderado; alto; “galeroso” (estilo entusiasmado).
- **Neutro:** interesse alto; interesse baixo.

**3) Uso de gírias e vocabulário recifense (exemplos)**

As frases devem usar e misturar gírias recifenses, tais como:

- **Interjeições e vocativos:** visse, oxe, ie é é?, bora ver, meu comparsa, meu irmão.
- **Gírias pejorativas/insultos regionais:** tabacudo, mungango, seboso/seboseira, pantim, galado, misera, disgraça, vai te fuder (contextualizado como insulto/expressão de irritação), tiração do carai, lascado.
- **Elogios e expressões positivas:** massa/massa demais, arretado, estribado, redondinho, ajeitadinho, cabuloso, bombando.

**4) Estrutura e fluidez**

- As frases devem variar bastante na estrutura.
- Não começar sempre com a mesma gíria.
- Misturar frases que: começam com ação/emoção; introduzem consequência do que aconteceu; apresentam diálogo ou comentário interno.
- Incluir gírias no meio da frase, não só no início.
- Refletir o modo de falar natural do Recife.

**5) Exemplos de frases desejadas (para referência no prompt)**

*Reclamação / agressivo (com CNV):*

- “Tu é muito tabacudo, visse? Deixou o serviço todo troncho e agora eu que vou ter que ajeitar essa seboseira.”
- “Isso me deixa bolado, meu irmão, porque ninguém respeita o combinado — sinto frustração / preciso de clareza.”
- “Vai te fuder com essa postura, disgraça, parece até que tu faz por pirraça.”

*Elogios:*

- “Boy, cê mandou massa demais nesse trampo estribado, ficou redondinho.”
- “Meu comparsa, tava cabuloso ver como tu resolveu aquilo arretado.”
- “Ficou massa meu comparsa, tu tira onda.”
- “Tá no esquema boy, é jogo.”
- “Bora ver meu irmão? O resultado ficou bombando de bom!”

*Neutro:*

- “Quero entender melhor essa ideia, visse, explica com calma.”
- “Tá certo então, depois a gente vê isso numa boa, meu comparsa.”

### Texto do segundo prompt (para colar no LLM)

```
Gere 120 frases no dialeto recifense para CSV, use variedade de tons (reclamação explícita/moderada, agressividade/xingamento, elogios moderado/alto/galeroso, neutro interesse alto/baixo). Inclua gírias recifenses como visse, oxe, meu comparsa, bora ver meu irmão, ie é é?, tiração do carai, vai te fuder (insulto), seboso, pantim, galado, massa, arretado, estribado, misera, desgraça.
Distribua gírias de forma natural e variada, não repetitiva, espalhadas ao longo das frases.
```

---

## 3. Prompt consolidado em uma linha (para colar)

Versão única, pronta para uso, que reúne quantidade, tamanho, tons e gírias.

```
Gere 120 frases no dialeto recifense para CSV, com 30–255 caracteres (até 300 quando necessário), variando tons: reclamação explícita/moderada, agressividade/xingamento, elogios moderado/alto/galeroso, neutro interesse alto/baixo. Use gírias recifenses como visse, oxe, meu comparsa, bora ver meu irmão, ie é é?, tiração do carai, vai te fuder (insulto), seboso, pantim, galado, massa, arretado, estribado, misera, desgraça, tabacudo, mungango, seboseira, redondinho, ajeitadinho, cabuloso, bombando. Distribua as gírias de forma natural e variada ao longo das frases, com estruturas diversas (não só no início), refletindo o falar natural do Recife.
```

---

## Observações (a partir dos insights)

- Após a geração, foi necessária **curadoria humana**: correção de acentos, pontuação e encoding (UTF-8), além de introdução de melhores frases para enriquecer a variedade linguística e o regionalismo.
- Um prompt genérico tende a não aprofundar no regionalismo do Recife/nordeste e pode gerar repetição, “alucinações” (palavras que ninguém usa) e variação linguística limitada; daí a importância da documentação e da curadoria para o dataset utilizado no TCC.
