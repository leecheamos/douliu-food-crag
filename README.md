# 斗六美食 CRAG × LINE

實驗室面試 demo 的 Python 專案。目標為斗六餐飲搜尋、評估與推薦，透過 LINE 對話操作。

目前可在本機 HTTP 聊天入口解析並保存多輪條件，對 31 筆餐廳來源資料執行硬條件篩選與 TF-IDF 候選排序。候選由本機硬條件覆核及 Gemini 模型評估，最多一次子類別擴展修正，再以固定格式產生至多 3 家有來源的推薦。LINE webhook 已實作（第 8 項，等待真實帳號驗收）。

## 啟動（Python 3.11 以上，PowerShell）

使用 Python 標準函式庫，不需第三方套件。模型評估需 GEMINI_API_KEY；沒有金鑰時仍可做本機檢索，但模型評估會明確停止，不冒稱通過。

```powershell
$env:PYTHONPATH = 'src'
python -m food_crag.server --env-file .env
```

預設網址 http://127.0.0.1:8000 。使用 Ctrl+C 停止。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

健康檢查回傳 status=ok、mode=gemini_evaluation、recommendation_ready=true（第 7 項起），以及 embedding 狀態（見下方「選用：Embedding 混合排序」）。

| 端點 | 行為 |
|---|---|
| GET /health | 200，模型評估流程模式；recommendation_ready=true；embedding 狀態 |
| POST /api/chat | 合法 JSON 含 user_id、message 時回傳 200，包含條件摘要、追問、檢索與評估紀錄、推薦回覆及 restaurant_ids |
| POST /webhooks/line | LINE webhook：驗證 X-Line-Signature，通過回 200 並在背景回覆；簽章錯誤 401，未設定 LINE 憑證 503 |

設定範例見 .env.example。程式預設只讀程序環境變數；加上 `--env-file .env` 才明確載入本機檔案，程序環境變數優先。檔案只支援 KEY=value（可用成對引號），不執行命令或展開變數；不要把金鑰貼到聊天或提交版本控制。如需改埠，啟動前設定 `$env:APP_PORT = '8001'`。此伺服器僅供本機使用。

## 驗證

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
```

測試實際啟動本機 HTTP server，驗證健康檢查、輸入驗證，多輪條件與使用者隔離，以及 LINE 未串接時不會假裝成功。沒有呼叫外部 API。

## 專案文件

- [系統架構圖與元件對照](docs/ARCHITECTURE.md)
- [開發步驟與進度](DEVELOPMENT_PLAN.md)
- [餐廳資料說明](data/README.md)

已完成原 mini_crag_tutorial 的沿用檢視，見 [原碼檢視報告](docs/REUSE_REVIEW.md)。TF-IDF 計分核心可移植，餐飲條件與評估規則需改寫；已在第 5 項移植餐飲檢索，原始 Mini CRAG 檔案未修改。

後續開發位置：`C:\DEV\CRAG\food\_CRAG`。

## 第 3 項：餐廳資料驗證與載入

正式資料見 [資料預覽](data/RESTAURANTS_PREVIEW.md)，含 DL-001～DL-032（DL-011 已併入 DL-010），共 31 家。原始草稿保留不改寫。

```powershell
$env:PYTHONPATH = 'src'
python -B -m food_crag.dataset
python -B -m unittest discover -s tests -v
```

預期資料檢查輸出 records=31、with_price=31、with_menu=3、with_hours=3、with_landmarks=1；目前共 57 項測試方法（包括資料、解析、對話狀態與實際本機 HTTP；部分方法涵蓋多個子案例）。Python API 為 `food_crag.dataset.load_restaurants()`，預設依模組位置定位資料，可從不同工作目錄載入。

`documented` 僅表示欄位與來源紀錄通過資料檢查，不表示店家現況、菜單或營業已向店家確認。所有價格及午晚餐供應皆未知，不能據此承諾符合預算或用餐時段。聊天端點回傳條件、追問及檢索候選；候選尚未評估，不是最終餐廳推薦。

## 第 4 項：條件解析與對話狀態（已由使用者接受）

Python `re`、同義詞表及 `dataclasses` 負責條件解析；記憶體字典以 `user_id` 隔離狀態，以鎖保護同一輪讀取／解析／寫入。不呼叫模型或網路服務。

| 輸入 | 處理方式 |
|---|---|
| 200 元以內、預算改成150、不超過200元 | 設定或替換預算上限 |
| 想吃麵、麵類、麵店 | 統一為麵食；預設替換前輪料理偏好 |
| 也想吃飯、加入牛排 | 追加可接受的料理；多種料理表示擇一皆可 |
| 不要火鍋和牛排 | 累加排除條件 |
| 取消排除火鍋、火鍋也可以 | 只移除火鍋排除，不影響其他排除 |
| 取消預算限制、取消料理限制、取消排除條件 | 清除指定欄位 |
| 午餐、晚餐、時段不限 | 設定或清除時段；不表示已確認店家供應 |
| 雲科大附近、斗六車站附近 | 正規化地標；不使用定位，不計算距離 |
| 取消地標限制 | 清除地標 |
| 重設、重新開始、清除條件、reset | 清除該使用者全部狀態 |

地區固定雲林縣斗六市；地標名稱目前支援雲林科技大學（雲科大）、斗六車站（斗六火車站）、雲林縣政府、鎮南國小站。能解析地標不代表資料集中已有相符店家。料理詞彙見 `src/food_crag/criteria.py`。

未提供的新欄位保留前輪；新料理預設替換料理偏好，排除偏好預設累加。衝突、未知條件、區間／小數／多個預算等不確定輸入會追問，**整輪不儲存任何修改**，並顯示原條件。例如花生過敏、素食、外帶、辣度、步行距離目前均不支援，會追問而不忽略。

合法聊天請求 HTTP 200，`status` 為 `criteria_updated`、`clarification_required` 或 `reset`；回傳 `criteria`、`questions`、`reply`，`restaurant_ids` 仍為空；第 5 項另由 `retrieval.candidates` 提供未評估候選。狀態僅在目前程序中保存，重啟清空；`user_id` 是本機呼叫者提供的識別碼，尚無正式登入或 LINE 身分驗證。

### 本次實際 HTTP 驗證流程

| 同一使用者輸入 | 狀態與結果 |
|---|---|
| 200元以內，不要火鍋 | 記錄預算200、排除火鍋 |
| 想吃麵 | 保留預算與排除，料理變麵食 |
| 預算改成150 | 預算變150，其餘保留 |
| 想吃火鍋 | 追問衝突；保留原有150元、麵食與排除火鍋 |
| 取消排除火鍋，想吃火鍋 | 明確移除火鍋排除，料理變火鍋 |
| 重設 | 清除該使用者條件；其他使用者不受影響 |

伺服器啟動後可由 PowerShell 呼叫（無須在第 4 項提供任何金鑰）：

```powershell
$payload = @{ user_id = 'demo'; message = '200元以內，不要火鍋' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/chat' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($payload))
```

變更 message 並使用相同 user_id 可追加條件。第 4 項的自動測試已代為執行，不要求使用者操作測試。沒有圖形 UI；本項展示入口為 HTTP API 與本頁對話範例。

## 第 5 項：餐廳檢索器（已由使用者接受；以下為當時檢索範圍）

已移植 `mini_crag_tutorial/retriever.py` 的 TF、平滑 IDF、cosine similarity、詞彙貢獻及已知查詢詞覆蓋率公式，以及 `tokenizer.py` 的中文 bigram；移除學制詞彙與資料路徑，改用餐飲同義詞。原 Mini CRAG 與論文程式皆未修改，沒有新增依賴。

實作：`src/food_crag/tfidf.py`、`retriever.py`；接入 `pipeline.py`，新增報告資料契約與 `tests/test_retriever.py`。目前只是檢索，不是第 6 項 CRAG evaluator。

### 流程與篩選界線

1. 啟動時載入資料並預建記憶體 TF-IDF 索引，每輪不重建。修改資料後須重啟服務。
2. 先檢查地區、料理分類、排除、預算、用餐時段及地標；不符與未知分開記錄。有已知不符時即剔除，即使其他條件未知也不列候選。
3. 對硬條件通過者計算相似度。零分不列候選，依分數降序、ID 升序處理同分；預設至多 10 筆候選，供後續評估使用。
4. 回傳每家篩選結果及原因、候選分數、命中詞／貢獻、已知查詢詞覆蓋率與原餐廳來源。分數和覆蓋率都不是推薦正確機率。

料理篩選僅對資料中已記錄的分類與已知上下類關係判斷，不把完整菜單或食材安全當成已確認。麵食可涵蓋牛肉麵、羹麵等已標記子類；只標麵食不能反推為牛肉麵，要求排除牛肉麵時也不能直接視為安全符合。料理／地標未知時不靠店名相似度補成符合。

預算依菜單照片的**主餐單品標價**判斷（2026-11-19 起）：最便宜的主餐 ≤ 預算即符合，超過則剔除，沒有價格資料為未知。小菜、湯、飲料、時價與合菜不算主餐；收服務費的店以含服務費金額記錄。這不是每人整餐費用，推薦回覆會註明「單品價不等於整餐費用」。

用餐時段使用明確的 meal_periods 與來源，不能由 opening_hours 推論供餐。地標只比對已記錄標記，不計算距離。營運現況仍未知，不宣稱即時營業。

排序文字使用累積的料理偏好與地標；沒有正向料理／地標時使用斗六市作為地區瀏覽排序，不代表餐廳品質比較。排除詞、價格更新句、來源附註與排除原因不作為正向排序依據。直接呼叫 `search(query, criteria, limit)` 時可指定查詢詞；空白或無詞彙命中的查詢不產生候選。

### HTTP 回傳與實際驗證

合法聊天仍為 HTTP 200，外層 status 保持第 4 項行為。條件更新後新增 `retrieval`：

| 欄位／狀態 | 意義 |
|---|---|
| query | 本次累積正向查詢 |
| candidates_found | 有硬條件通過且詞彙分數大於零的候選 |
| insufficient_data | 零候選，仍有店家因必要資訊未知而無法確認 |
| no_results | 零候選，已知不符或無詞彙命中；不自動放寬 |
| empty_dataset | 注入空資料集的明確結果；正式啟動仍由資料驗證器檢查 |
| candidates | 原餐廳資料、score、matched_terms、evidence、query_coverage |
| decisions | 每家 ID、eligible／excluded／unknown／no_lexical_match 及原因 |
| total / eligible_count | 總筆數／硬篩通過數；後者包括詞彙零分者，未必等於回傳候選數 |

追問或重設時 `retrieval=null`，不觸發檢索。`restaurant_ids` 仍為空，避免把檢索候選當作最終推薦。沒有評估器、自動重試、外部搜尋或推薦文字生成。

本次使用正式資料經實際本機 HTTP 驗證：

| 輸入 | 檢索結果 |
|---|---|
| 想吃麵，不要火鍋 | 5 筆候選：DL-017、DL-027、DL-032、DL-002、DL-028 |
| 同一使用者追加「200元以內」 | insufficient_data，零候選，5 家麵食候選價格未知；原條件保留 |
| 再輸入「取消預算限制」 | 恢復相同 5 筆；由使用者明確解除限制 |
| 另一使用者「鎮南國小站附近」 | DL-025；其餘 31 家地標資訊未知 |
| 另一使用者「午餐」 | insufficient_data，32 家供餐時段未知 |

首筆麵食候選為烹小鮮麵館，分數約 0.5868；分數只用於目前資料的詞彙排序。以上不是最終推薦或即時價格／營業驗證。

### 驗收證據

`python -B -m unittest discover -s tests -v`：40 項測試方法通過，其中第 5 項新增 15 項。涵蓋數學公式、排序、硬條件、未知／超支／區間價格、來源缺漏、上下料理類別、空資料與零分、無效 limit、索引重用、同分穩定排序、正式資料，以及真實本機 HTTP 多輪／隔離／零結果。人工價格與供餐時段測試資料只存在測試檔，沒有寫入正式資料集。

本項技術驗證及 AI 需求自查通過，已由使用者以「測試OK」明確接受。無圖形 UI，展示入口為本頁結果與 `/api/chat`，不要求使用者代跑測試。第 6 項未開始。

## 第 6 項：Gemini 評估與有限修正（已由使用者接受）

本節為最新執行行為，取代第 5 項當時「尚無 evaluator／每輪只檢索一次」的描述。第 7 項最終推薦未開始。

### 已設定的串接方式

- REST：`https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent`。
- 認證使用 `x-goog-api-key` header，不放在網址、模型提示或 HTTP 日誌。
- 本次採 `gemini-3.5-flash-lite`，真實帳號已成功呼叫。先前建議的 3.8 可列出模型，但生成曾在 20／45 秒及最小 15 秒請求逾時，因此改用實測可用模型；金鑰未更動。
- 透過 JSON Schema 指定輸出格式，回來後仍由本機檢查型別、狀態一致性、候選 ID 及改寫白名單，不把 JSON 格式視為事實保證。
- `.env` 不自動載入，使用本節指令明確指定；模型欄位空白時預設為 `gemini-3.5-flash-lite`，GEMINI_TIMEOUT 預設 20 秒，可設定大於 0、至多 60 秒。

```powershell
Set-Location -LiteralPath 'C:\DEV\CRAG\food\_CRAG'
$env:PYTHONPATH = 'src'
python -B -m food_crag.server --env-file .env
```

`.env` 主要欄位（不要將真實金鑰放入文件）：

```dotenv
GEMINI_API_KEY=只在本機填入你的金鑰
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_TIMEOUT=20
```

若要單獨驗證真實服務，已有命令 `python -B -m food_crag.verify_gemini --env-file .env`。此命令會呼叫外部 API；一般 unittest 不呼叫真實 API。本次已由 AI 代為執行，使用者無須代跑。

### 評估與停止規則

1. 本機重新比對候選 ID 與原資料、硬條件、TF-IDF 分數、命中詞貢獻及基本來源。偽造／缺證據或不符條件時，Gemini 無權覆蓋。
2. 零候選且仍有必要資訊未知：`clarify / insufficient`，不呼叫模型；零候選且已知不符或無詞彙命中：`stop / no_results`。
3. 有可核對的候選才交 Gemini，內容是結構化條件與候選證據，不含 API key 或 user_id，不讓模型瀏覽網址或執行工具。
4. 模型可回傳 accept、retry、clarify、stop。accept 必須是 sufficient、至少一個合法候選 ID、rewrite_query 為空；不滿足即停止，不替模型修補矛盾回覆。
5. retry 只能選本機提供的子類別擴展查詢：例如「麵食」擴展為「麵食 牛肉麵 鍋燒意麵 拉麵 義大利麵 羹麵」、「飯食」擴展為「飯食 肉燥飯 豬腳飯 菜飯 蛋飯」。子類別比原需求更窄，所有搜尋條件完全保留，排除的料理不加入擴展；無子類別的料理（如火鍋、熱炒）不提供修正。不能調高預算或移除排除偏好。
6. 每輪最多一次修正，即至多兩次檢索與兩次模型呼叫；再次要求 retry 會停止。缺價格、未知供餐時段等不能靠擴展查詢補足，不自動重試 API 錯誤。
7. 逾時、HTTP 錯誤、阻擋／截斷回覆、錯誤 JSON、捏造 ID 或非法改寫皆停止，沒有假裝模型通過的本機備援。

同義詞（麵食／麵類）在詞彙索引中本來就正規化，互換不會改變結果，因此 2026-11-19 改為子類別擴展。擴展只改變候選排序，不增加候選數（料理篩選已含子類別店家）。真實 Gemini 實測：「想吃飯」第 1 輪要求修正，擴展後首位由 DL-010（菜飯）改為 DL-026（肉燥飯），第 2 輪 accept；不保證每次模型都會選擇修正。這是簡化 CRAG 流程，不是論文完整重現。

### HTTP 證據（2026-11-03）

真實本機 HTTP 加真實 Gemini：

| 輸入 | 評估結果 | 真實模型呼叫 |
|---|---|---|
| 想吃麵，不要火鍋 | accept / sufficient，接受 DL-017、DL-027、DL-028，約 1.95 秒 | 是，1 次 |
| 同一使用者追加 200 元以內 | clarify / insufficient；正式資料價格未知，條件保留 | 否 |
| 另一使用者：想吃熱炒，不要台菜 | stop / no_results；依目前分類無符合候選 | 否 |

單獨真實 API 驗證亦通過。一次早期模型回覆同時填入 accept 與改寫查詢，被本機拒絕；補強提示後重新驗證通過，沒有放寬輸出驗證規則。未據此保證每次呼叫結果相同或服務永遠可用。

新增 HTTP 欄位：`evaluation`（decision、evidence_state、accepted_ids、reasons、rewrite_query、model_called），以及 `attempts`，每輪包含查詢、原條件、完整檢索與評估結果。`model_called` 表示曾嘗試外部呼叫，成功須再看 decision/state；未設定金鑰不標為已呼叫。原 `retrieval` 保留供檢查，不表示其中所有候選都被模型接受。`restaurant_ids` 仍為空，第 7 項才產生最終推薦。

### 驗收與範圍

- `python -B -m unittest discover -s tests -q`：57 項測試方法通過；本項新增 17 項（含 2026-11-19 連線中斷與子類別擴展測試），含多個子案例。
- 測試涵蓋本機否決、模型欄位一致性、假 ID／內容／分數／詞彙證據、未知價格、一次修正、上限、429／逾時／無效回覆、明確 env 載入及 HTTP 流程；模型與網路錯誤測試是模擬證據。
- 真實服務證據：金鑰可取得模型清單、3.5 Flash-Lite 文字與結構化評估成功，以及上述 HTTP 三分支。
- 本項技術驗證與 AI 需求自查通過，2026-11-15 已由使用者接受（2026-11-19 已修正連線中斷例外未攔截問題）；未提交、推送或部署，未開始第 7 項。

官方參考：[金鑰設定](https://ai.google.dev/gemini-api/docs/api-key)、[generateContent REST](https://ai.google.dev/api/generate-content)、[結構化輸出](https://ai.google.dev/gemini-api/docs/generate-content/structured-output)。

## 第 7 項：有依據的推薦回覆（方案 A，2026-11-20 已由使用者接受）

本節取代前面各項「restaurant_ids 為空、不產生最終推薦」的描述。

- 實作：`src/food_crag/recommendation.py`。`TemplateReplyGenerator` 只用本機資料填固定格式，不呼叫模型；`verify_recommendation` 在回覆送出前檢查。
- 推薦：取 Gemini 接受且在本輪檢索候選中的餐廳，依檢索排序至多 3 家，不足不硬湊。每家附推薦理由（資料分類符合的料理、未含排除料理、地標標記）、價格（無來源時寫「資料未知」）、地址、地圖連結、資料日期與來源，最後附「未向店家確認」提醒。
- 非推薦情況：資料不足列出缺少的資訊與家數並提示可明確放寬；零結果說明未放寬條件；模型失敗或未設定金鑰時回「推薦評估暫時無法完成，請稍後再試」，不列出未經評估的餐廳，也不顯示錯誤代碼。
- 驗證：推薦的 ID 須存在於資料集、被 Gemini 接受、在檢索候選內、不重複且至多 3 家，店名與地址須逐字出現在回覆中。生成器拋錯或驗證失敗時改用固定格式，再失敗則回不可用訊息。
- 方案 B 預留：`RecommendationPipeline(reply_generator=...)` 可替換為模型撰寫的生成器，輸出 `Recommendation(reply, restaurant_ids)`，仍經同一驗證並以方案 A 備援。

### 實際 HTTP＋真實 Gemini（2026-11-19）

| 輸入 | restaurant_ids | 回覆重點 |
|---|---|---|
| 想吃麵，不要火鍋 | DL-017、DL-027 | 推薦 2 家，價格資料未知，附地址、地圖、來源 |
| 同一使用者追加 200元以內 | 空 | 資料不足：缺少有來源的每人餐費資料（5 家） |
| 想吃熱炒，不要台菜 | 空 | 沒有同時符合條件的店家，未放寬條件 |
| 想吃飯 | DL-026、DL-005、DL-010 | 經一次子類別修正後推薦 3 家 |

`python -B -m unittest discover -s tests -q`：65 項測試方法通過，本項新增 8 項（`tests/test_recommendation.py`）。正式資料價格皆未知，所以目前推薦都不會顯示價格；有價格的格式只用測試資料驗證。

## 選用：Embedding 混合排序（2026-11-19 追加，已由使用者接受）

預設關閉，行為與原本相同（`score == lexical_score`，`ranking=lexical`）。啟用後以 Gemini Embedding 計算語意分數，與 TF-IDF 排名用 RRF（k=60）融合，**只重新排序**：候選仍須通過硬條件且有詞彙命中，不新增召回、不放寬條件。`semantic_score` 不是條件證據。

### 啟用方式

1. 在 `.env` 設定 `EMBEDDING_MODEL=gemini-embedding-001`（空白即關閉）。
2. 建立文件向量快取（會呼叫外部 API，32 家店）：

```powershell
$env:PYTHONPATH = 'src'
python -B -m food_crag.embedding --env-file .env
```

3. 照常啟動伺服器；查詢向量在記憶體快取。查詢向量呼叫失敗時該輪退回 TF-IDF（`ranking=lexical_fallback`）。

快取位置：`data/embeddings/<模型名稱>.json`。以 sha256 比對每家店的檢索文字（店名、縣市、行政區、料理分類、菜單摘要、地標），**`data/restaurants.json` 有變更就必須重建**，否則自動退回 TF-IDF。

### /health 的 embedding 狀態

| 值 | 意義 |
|---|---|
| disabled | 未設定 EMBEDDING_MODEL |
| hybrid_rrf | 快取有效，啟用混合排序 |
| cache_missing | 找不到快取檔，需執行建立指令 |
| cache_stale | 餐廳資料已變更，快取過期，需重建 |
| cache_model_mismatch | 快取的模型與設定不同 |
| cache_invalid | 快取格式損壞 |

除 hybrid_rrf 外皆只用 TF-IDF 排序，不會讓服務啟動失敗。

### 回傳欄位與驗證

- `Candidate` 新增 `lexical_score`、`semantic_score`；`RetrievalResult` 新增 `ranking`（lexical / hybrid_rrf / lexical_fallback）。
- 評估器以 `lexical_score` 覆核 TF-IDF 證據；hybrid 模式下重跑檢索，候選須完全一致，否則 stop/error。
- 測試：`tests/test_embedding.py` 新增 15 項，全部共 80 項通過。

## 第 8 項：LINE 聊天機器人（2026-11-20 已由使用者接受）

- `.env` 需填 `LINE_CHANNEL_SECRET`、`LINE_CHANNEL_ACCESS_TOKEN`；啟動 `python -B -m food_crag.server --env-file .env`，/health 的 `line_configured` 為 true 表示已讀到憑證。
- 流程：LINE 呼叫 `POST /webhooks/line` → 以 Channel secret 對原始內容做 HMAC-SHA256 比對 `X-Line-Signature`（錯誤回 401、不處理）→ 立即回 200 → 背景執行緒（最多 4 個）跑推薦流程 → 以 reply token 回覆。等待時在一對一聊天顯示輸入中動畫。
- 使用者隔離：對話狀態以 `line:<userId>` 為鍵，與 `/api/chat` 的 user_id 分開。
- 重送事件：以 `webhookEventId` 去重（記憶最近 2048 筆），LINE 重送的事件仍回 200 但不重複回覆。
- 非文字訊息回覆使用說明；follow 等事件交給 LINE 官方帳號的歡迎訊息。回覆依換行切段，最多 5 則、每則 5000 字。
- 只用 reply API（免費），不用 push；回覆失敗（例如 reply token 過期）只記錄錯誤代碼。日誌不記錄訊息內容、使用者 ID 或憑證。
- 限制：狀態與去重紀錄存在記憶體，重啟清空；本機 server 需透過 HTTPS 通道或部署（第 9 項）才能讓 LINE 連入。
- 測試：`tests/test_line.py` 11 項（簽章、重送、隔離、非文字、錯誤處理、真實本機 HTTP＋真實推薦流程）；LINE API 以 mock 驗證，共 91 項測試通過。

### 用 Docker 連上 LINE（本機 demo）

**一鍵啟動：** `powershell -ExecutionPolicy Bypass -File .\demo.ps1`，會自動啟動容器、取得通道網址、用 LINE API 更新 Webhook URL 並測試連線。`docker compose logs -f app` 可看每次回覆的延遲秒數（不含訊息內容）。以下為手動步驟。

- `docker compose up --build`：啟動兩個容器。`app` 是推薦程式，不對電腦開任何 port；`tunnel` 是 Cloudflare 臨時通道（`--protocol http2`），也是唯一的對外入口。
- `docker compose logs tunnel`：找 `https://….trycloudflare.com`，把 `<網址>/webhooks/line` 填到 LINE Developers 的 Webhook URL。通道每次重啟都會換網址。
- `docker compose down`：停止，公開網址隨之失效。
- 金鑰不打包進映像檔（`.dockerignore` 排除 `.env`），執行時由 compose 從 `.env` 讀入。容器以非 root 使用者執行，並有 /health 健康檢查。
- compose 設定 `CHAT_API_ENABLED=false`，公開時 `/api/chat` 回 404，只開放 `/webhooks/line` 與 `/health`。

### 推薦回覆格式（2026-11-20 起）

口語開場「想吃麵食、100 元以內的話，我推薦這 3 家：烹小鮮麵館，主要賣豬油拌飯、乾拌麵和燒餅；…」，接著每家四行：店名、地址、主餐價格範圍、Google Maps 網址，最後一行提醒單品價不是整餐費用、出發前確認營業。「主要賣」來自依菜單照片整理的 menu_summary。

菜名（2026-11-21）：每家另有 menu_items（菜單照片上的全部主餐）。使用者指定菜名時，只推薦店名、招牌或 menu_items 有賣該菜名的店：品項字面包含菜名即算；菜名以飯、麵、丼、麵線、便當結尾且主料至少兩字時，品項含主料並以同一主食結尾也算（「咖哩豬排蛋包飯」算豬排飯；「排骨飯」不算；「咖哩炸飯糰」不算咖哩飯；「炒飯」維持字面比對），開場改為「…有賣滷排骨飯」；有賣但被預算等條件擋掉時說明「有 N 家賣X，但不符合你的其他條件（原因）」；都沒有時說明「目前資料裡沒有找到賣X的店，先推薦相近的…」。有賣該菜名的店視同符合料理分類。限制：預算以店內最便宜主餐判斷，不是該道菜的價格；同時指定菜名與預算時，回覆會加註「預算是用店內最便宜的主餐判斷，不一定是X的價格」。價格依據與資料來源不顯示在聊天回覆，但仍在資料集與 /api/chat 回傳中。

### 自由輸入（2026-11-20 起）

- 規則解析優先；規則讀不懂整句（有未知字詞或沒抓到任何條件）時，才交給 Gemini（`llm_criteria.py`）把整句話＋前一輪條件轉成完整的新條件。衝突、多個預算等規則刻意追問的情況不交給模型。
- 模型輸出只當資料，由程式檢查：料理、排除、地標、時段只能用固定詞彙；**菜名與新的預算數字必須出現在使用者原句**（支援國字數字，如「一百塊」），否則丟棄；料理與排除衝突改為追問。過敏、距離、外送等不支援的需求由模型回追問，不會默默忽略。API 失敗或輸出無效時沿用規則的追問。
- 新增 `dishes`（菜名）條件：只加入排序查詢，不做硬篩選；送給評估模型時以 `dish_preferences` 提供，並明示菜單摘要不是完整菜單，不能因沒列出某道菜而拒絕。
- 回覆格式不變；開場改說「想吃豬排飯的話…」。
- 限制：Gemini 免費額度有每分鐘次數上限，短時間連續大量測試可能暫時失敗（回「推薦評估暫時無法完成」）；同一句話的評估結果仍可能因模型判斷而不同。
- 時段（午餐／晚餐，2026-11-20 使用者選擇方案 A）：改為偏好，不再因供餐時段未知而擋下；只有資料明確標示不供應時才剔除。回覆最後一行改為「是否有供應晚餐尚未確認，出發前請先查營業時間」。評估模型以 `meal_preference` 取得，並明示不得因時段未知而拒絕。
