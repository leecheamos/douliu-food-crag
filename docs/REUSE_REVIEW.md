# 原系統沿用檢視

檢視日期：2026-09-07。

## 第 5 項移植更新（2026-11-02）

目前開發位置為 `C:\DEV\CRAG\food\_CRAG`。原 Mini CRAG 的 TF、平滑 IDF、TF-IDF、cosine、token 貢獻與覆蓋率已移植至 `src/food_crag/tfidf.py`；中文 bigram 改用餐飲同義詞。`retriever.py` 已加入結構化硬篩、預建記憶體索引、來源保留與逐候選篩除原因。同分以 ID 排序，未知硬條件與零分不列候選。

本次第 5 項共新增 15 項測試方法，連同原有測試 40 項通過，另完成真實本機 HTTP 正式資料驗證。價格與時段正向案例使用合成測試資料，不是店家事實。檢索後尚無 evaluator、修正重試或最終推薦，第 6 項未開始。未修改原 Mini CRAG 或論文程式，沒有新增依賴。

以下內容為第 2 項的歷史檢視紀錄；其中「尚未移植／尚未實作」請以本更新及 DEVELOPMENT_PLAN.md 的現況為準。

## 原碼位置與結論

- 主要沿用來源：`C:\Users\USER\OneDrive\桌面\雲科\論文\CRAG\mini_crag_tutorial`。
- 論文實作參照：同層 `CRAG_code`。
- 餐飲開發目錄：`C:\Users\USER\OneDrive\桌面\雲科\論文\CRAG\food\_CRAG`。與同層現有 `food_CRAG` 是不同路徑。

已實際閱讀 app.py、retriever.py、tokenizer.py、evaluator.py、answer_generator.py、chunk_documents.py、evaluate.py 與 README；另檢視 CRAG_code 的依賴及 CRAG_Inference.py 模型與分流結構。

原系統實際為大學、碩士、博士選課規定的離線問答，不是僅限碩士招生問答。它使用字元 bigram、TF-IDF、cosine similarity 與規則 evaluator；沒有 Gemini API，也沒有神經網路 embedding。TF-IDF 是稀疏詞彙向量，不宜稱為能理解語意的 embedding。

第 2 項完成的是原碼檢視與骨架，不是餐飲檢索器實作。以下「可沿用」表示已確認可移植，尚未把原函式搬入餐飲執行流程。

## 逐項沿用對照

| 原檔案／函式 | 結論 | 餐飲接入與必要調整 |
|---|---|---|
| retriever.py：calculate_tf、calculate_idf、cosine_similarity | 可沿用數學核心 | 第 5 項抽成獨立稀疏檢索模組，不保留課程資料路徑 |
| retriever.py：calculate_tfidf、vectorize_query、build_tfidf_index | 小幅改寫 | 改成餐廳文字索引、可注入 tokenizer；維持 ID 對應；新增空資料與未知詞測試 |
| retriever.py：retrieve | 沿用計分與排序流程 | 先套用硬條件，再排序；避免 zip 靜默忽略長度不一致；驗證 top_k；零分不能直接當推薦依據 |
| retriever.py：explain_match、calculate_query_coverage | 可沿用診斷核心 | 呈現命中詞與覆蓋率，支援面試展示；覆蓋率不是推薦正確機率 |
| tokenizer.py：tokenize | 保留 bigram 與英數切詞 | 替換學制同義詞表；否定詞、預算與時間另交條件解析器，不能靠詞彙相似度判斷 |
| evaluator.py：calculate_query_diagnostics | 可沿用診斷思路與計算 | 保留完整問題覆蓋率，避免只看已知詞造成覆蓋率偏高 |
| evaluator.py：detect_programs、detect_domain_terms | 必須改寫 | 改為地區、料理、預算、時段與對話脈絡判斷；「換一家」等追問不能只靠餐飲關鍵字判定領域 |
| evaluator.py：evaluate_retrieval、process_flag | 沿用評估與三分流概念 | 餐飲需逐候選驗證；不能把排名前二接近視為必然模糊，因多家同樣適合是正常結果 |
| evaluator.py：各 threshold 常數 | 不直接沿用 | 原值校正於 10 條教學規定，需用餐飲開發集重新校正，再用獨立測試集檢驗 |
| answer_generator.py：answer_query | 沿用檢索→評估→回答的編排 | 改用餐飲資料契約，加入使用者狀態及有限重試 |
| answer_generator.py：generate_answer | 改寫模板 | 改成餐廳推薦、價格證據、來源與日期；可接 Gemini，保留模型失敗時的模板回覆 |
| chunk_documents.py：extract_strips_from_psg | 保留供長資料擴充 | 第一版每家餐廳一筆結構化記錄；不直接用固定字數拆散地址、價格及來源關係 |
| evaluate.py：evaluate_case、percentage | 沿用評估框架 | 規則 ID 改餐廳 ID；新增條件符合率、未知資訊與多輪測試；Hit@3 不等於全部推薦都正確 |
| app.py：build_system | 沿用啟動時建立索引的概念 | CLI 改為後端＋LINE，新增簽章驗證、事件去重與狀態隔離 |

## 分流映射注意事項

原 Mini CRAG 的 Correct／Ambiguous／Incorrect 是證據狀態；骨架的 accept／retry／clarify 是後續動作，不能直接一對一替換。

- 證據足夠且符合硬條件：可推薦（accept）。
- 缺少必要資訊：追問（clarify）。
- 搜尋詞可改善且未達上限：保留硬條件重新檢索（retry）。
- 超出範圍、零結果或無法補足證據：明確說明；需要額外的結果狀態，不應無限重試。

第 6 項應將「證據狀態」與「動作」分開建模，並補足無結果／超出範圍的終止動作。

## Gemini 與原始 CRAG 的取捨

建議第一版先沿用本機 TF-IDF 作為可解釋的檢索基線，結合結構化條件；Gemini 用於自然語言解析及推薦文字，規則仍負責硬條件。神經網路語意 embedding 保留為可選擴充，不是本機檢索的必要依賴。

CRAG_code 的推論程式使用 T5ForSequenceClassification 與 vLLM，其依賴包含 torch、CUDA 相關套件及大型模型。這份面試 demo 不直接移植整套執行環境，也未下載模型或執行原始大型模型實驗。

原 Mini CRAG 沒有實際的查詢重寫迴圈、外部搜尋、Gemini、LINE 或連續對話狀態；這些不能宣稱已沿用或已完成。

## 驗證紀錄

使用 Python 執行原 mini_crag_tutorial/evaluate.py（停用 bytecode 寫入，未修改原始碼）：

- 三分流正確率：15/15。
- Retrieval Hit@3：10/10。
- Correct 引用正確率：5/5。

上述題目是曾用於調整規則的開發集，不代表未知問題或餐飲資料上的效果。

餐飲骨架另以 tests/test_scaffold.py 檢查實際本機 HTTP 行為：健康檢查、非法輸入、未實作聊天及未串接 LINE，共 4 項。尚無餐飲推薦準確率結果。
