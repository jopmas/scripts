Atualize a planilha de monitoramento de cenários Mandyoc com o estado que está no JSON da entrada padrão.
Ele foi gerado agora por collect_status.py (pastas locais dos cenários + jobs do usuário no hypatia).

## A planilha

- Google Sheets "Monitor_Scenarios", spreadsheetId 1yj0JtUXj8szrJiXnG5B_H0eIiiH2BoYmyOhlqxwIrTQ
  (https://docs.google.com/spreadsheets/d/1yj0JtUXj8szrJiXnG5B_H0eIiiH2BoYmyOhlqxwIrTQ), pelo conector Google Sheets.
- Cada aba é uma tabela com cabeçalho na linha 1 e um cenário por linha; a coluna A é "Cenário".
  Ache as colunas pelo nome no cabeçalho, nunca pela letra: o usuário pode ter movido ou criado colunas.
- O usuário edita esta planilha à mão. Leia cada aba (get_values) antes de escrever nela e escreva só nas
  células indicadas abaixo, com valores RAW (números como números).

## O que fazer

Para cada cenário do JSON (campo `tab` = nome da aba, `name` = valor da coluna Cenário):

1. Se a linha existe, atualize somente estas células, e só quando o valor for diferente do que está lá:
   - "Status" ← `status`;
   - "Job ID (hypatia)" ← `job_id` (número), se não for null;
   - "Tempo atingido (Myr)" ← `time_reached_myr`, arredondado a 1 casa (3 casas se for menor que 1), se não for null
     e se a aba tiver essa coluna.
   Não mexa em nenhuma outra célula: Comments, Comparison e o resto são do usuário.
2. Se a linha não existe e o cenário tem `update_only` = true, não faça nada (não crie a linha). Se a linha não existe
   nos outros casos, escreva-a na primeira linha vazia logo abaixo da tabela da aba:
   Cenário = `name`, Status, Job ID, "Tempo total (Myr)" = `time_max_myr`, Tempo atingido, Pasta = `folder`,
   "Grupo" = `group` (se a aba tiver essa coluna) e as demais colunas a partir de `columns` (as chaves são os nomes
   das colunas). Colunas da aba sem valor no JSON ficam vazias. Em Comments escreva "Adicionado pelo attplan.".
   Se o filtro básico da aba não cobrir a linha nova, estenda-o.
3. Se a aba não existe, crie-a copiando a aba "compression" (duplicateSheet), apague as linhas de dados da cópia,
   renomeie-a para `tab` e escreva as linhas dos cenários dela.

Depois:
- Em "Tempo atingido (Myr)", deixe o texto vermelho (#c62828) quando o status não for "Rodando" nem "Na fila" e o
  tempo atingido for mais de 2 Myr menor que o tempo total; volte à cor padrão quando isso deixar de valer.
  Mude só as células em que isso muda.
- Linhas da planilha sem cenário no JSON: não mexa nelas; só cite-as no resumo.
- Se `hypatia` não for "ok", atualize apenas com os dados locais e avise no resumo.

Junte as escritas em poucas chamadas (batchUpdate / updateCells). Não peça confirmação: este é um comando
automático do próprio usuário, que pediu essas atualizações.

## Resposta

Termine com um resumo curto em português, em texto simples (sem tabelas): o que mudou em cada aba
(cenário: valor antigo → novo), linhas novas e abas novas, e avisos. Se nada mudou, diga só isso.

## Exportação (obrigatória, sempre por último)

Depois do resumo, leia de novo todas as abas, na ordem da planilha, cada uma até a última linha e a última coluna
com dados, e escreva uma linha com exatamente @@EXPORT@@ seguida, na linha seguinte, de um único objeto JSON
compacto (sem bloco de código, sem nada depois):
{"sheets":[{"name":"<aba>","rows":[[<células da linha 1>],[<linha 2>],...]}]}
Cada linha tem o mesmo número de células (complete com null). Cada célula é o valor (número como número,
texto como texto, vazia → null). Um script lê esse JSON para salvar a cópia .xlsx no iCloud do usuário.
