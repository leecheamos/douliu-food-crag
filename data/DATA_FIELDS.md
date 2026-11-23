# 正式資料與原始草稿的欄位契約

2026-11-02：正式資料為 `restaurants.json`，共 32 筆。以下「草稿」章節保留 2026-09-07 的欄位沿革；當時的「未匯入／未建立驗證」已由本節取代。

- 正式資料 `review_status=documented`：欄位與來源紀錄通過結構檢查，不表示已確認現況或可滿足全部推薦條件。
- `operating_status` 仍為 `unknown`；資料沒有任何一家被標為當下營業中。
- 未知價格使用 null、未知菜單摘要使用空字串、未知時段使用 null、未知地標使用空陣列；不把未知當成符合條件。
- `checked_at` 是逐來源查閱日期的最大值，不是店家確認日期；人工提供且未獨立查閱時可為 null。新來源不覆寫原人工 reported_at 或 visited_at。
- 同一 URL 可以有不同日期／方法的來源觀察，`source_urls` 本身去重。`page`、`search_index`、`manual_entry` 分別記錄，直接頁面無法擷取的內容不升級為 page 證據。
- `source_date_kind` 支援 published、updated 及舊來源無法區別發布／更新的 published_or_updated；未知留 null。
- 新增來源類型 recruitment_platform、business_directory；企業招募／工商登記不能證明當日營運。
- 2026-11-19 起 `price_min_twd`／`price_max_twd` 改為**主餐單品標價範圍**（最便宜到最貴的主餐，不含小菜、湯、飲料、甜點、白飯、時價及多人合菜），**不是每人整餐費用**。另收服務費者以含服務費金額記錄；鍋物須另選湯底或主食者以最低組合計。預算判斷：最便宜主餐 ≤ 預算即符合，否則剔除；無價格為未知。
- 有數字價格必須兩端都有值，並同時有 price_evidence 文字及對應來源 supports。價格來源是使用者提供的菜單照片（manual_entry，note 記錄 `菜單/` 圖檔路徑），未獨立查核、拍攝日期多半未知。
- 非空的菜單、時段與地標需有對應 supports。料理仍是人工分類，不是完整菜單或飲食安全保證。
- `Restaurant` 明確映射核心欄位與 sources、notes、map_search_url、meal_periods、operating_status、review_status；列表轉為 tuple，來源附加欄位保留在來源物件中。
- 正式載入器 `food_crag.dataset.load_restaurants()` 會先驗證再載入。預設資料路徑依模組位置決定，不依賴啟動時工作目錄。

## 原始草稿欄位沿革（2026-09-07）

# 餐廳草稿欄位說明

此階段先定義資料格式，不建立驗證程式或匯入流程。JSON 使用 UTF-8；每筆是一家特定門市，ID 一經使用便固定，不隨排序變動。

| 欄位 | 格式與用途 |
|---|---|
| id / name | DL-001 格式的識別碼／店名；分店需區別 |
| county / district / address | 雲林縣／斗六市／來源記載地址；有差異記錄在 notes |
| cuisines | 人工分類陣列，依來源菜色或店名整理，不表示完整菜單或飲食保證 |
| menu_summary | 招牌品項（3 樣左右，依菜單照片），回覆「主要賣…」使用 |
| menu_items | 菜單照片上的全部主餐名稱（不含小菜、湯、飲料）；空陣列表示未知。用於檢索文字與菜名比對（`retriever.sells`：字面包含，或主料＋同一主食結尾），非空時來源 supports 須含 menu_items |
| menu_prices | 選填。`{菜名: 價格}`，菜名必須出自 menu_items，價格取菜單照片上最小份的單品價（正整數）；依菜單順序。有預算時，店裡要有一道「符合類型且在預算內」的菜才推薦，回覆也只列這些菜（最多 3 道）。套餐或鍋物組合（DL-020、023、024、029、030）不逐道標價，維持用 price_min_twd 判斷；Google 資料沒有此欄位。來源 supports 須含 price_evidence |
| price_min_twd / price_max_twd | 每人餐費數值或 null；本稿全為 null，不拿單品或禮盒價冒充餐費 |
| price_evidence | 價格依據或 null |
| opening_hours | 來源列示時段文字或 null，不等於即時營業狀態 |
| landmarks | 已有證據的地標陣列；尚未整理使用空陣列 |
| source_urls | 本筆可用來源網址；只有人工來源且無網址時使用空陣列 []，不填虛構連結 |
| checked_at | 本次來源查閱日期 YYYY-MM-DD，時區 Asia/Taipei；不是現況確認日期。人工來源尚未獨立查核時為 null，收到日期另記 reported_at |
| sources | 逐來源的 publisher、url、source_type、checked_at、access_method、source_date、source_date_kind、supports；人工來源另含 reported_at、visited_at、note |
| sources.access_method | page 表示開啟頁面；search_index 表示只取得搜尋索引內容，仍須複核；manual_entry 表示人工輸入 |
| sources.source_date | 頁面明示發布／更新日期；未知為 null，不以搜尋引擎爬取日代替 |
| sources.supports | 該來源支持的欄位；reference_or_conflict_note 表示補充或差異紀錄 |
| map_search_url | 根據店名及地址產生的地圖搜尋 URL，不作為查證來源 |
| meal_periods | 午晚餐供應；本稿為 null，避免把料理或時段推論成已確認供應 |
| operating_status | 本稿一律 unknown，尚未向店家確認營運現況 |
| review_status | 本稿一律 draft；不是可直接推薦的驗收資料 |
| notes | 搬遷、異名、舊資料、平台狀態等待查事項 |

來源類型：restaurant_official（店家官網）、government（政府）、media（媒體）、blog（個人食記）、partner（合作機構）、delivery_platform（外送平台）、corporate_document（企業公開文件）、personal_recommendation（人工推薦／提供）。類型不自動等於最新或正確。

資料沿用既有 Restaurant 核心欄位；sources、review_status 等為草稿附加欄位，後續匯入需顯式映射，不可直接 Restaurant(**record)。本階段不修改 domain.py。

待查看後續辦：補查地址差異與營運現況、最新菜單價格及用餐時段；建立資料驗證程式；完成正式匯入。第 3 項保持未完成。

## 人工來源

沒有官網的餐廳仍可建立草稿。publisher 如實記錄提供者或代稱，本次記為「使用者提供」，不推測是學長或店家。

| 欄位 | 人工來源規則 |
|---|---|
| sources.url | 無網址填 null；對應 source_urls 為 [] |
| sources.reported_at | 收到資料日期，與查核日分開 |
| sources.visited_at | 提供者實際用餐日期；未提供填 null |
| checked_at / sources.checked_at | 尚未獨立查核填 null，不以輸入日期替代 |
| sources.note | 保存提供方式與尚待查證的情況 |
| sources.supports | 表示人工提供涵蓋哪些欄位，不表示內容已獨立確認 |

本次新增記錄：

| ID | 店名 | 分類 | 人工提供地址 |
|---|---|---|---|
| DL-031 | 左撇子早午餐 | 早午餐 | 雲林縣斗六市龍潭里八德路27號 |
| DL-032 | 勞碌命鍋燒意麵 | 麵店 | 雲林縣斗六市鎮南里中山路240號 |

兩筆收到日期為 2026-09-07；價格、營業時間與實際用餐日期皆未知，review_status 保持 draft。日後匯入需處理 checked_at 可為 null 的情況，既有 domain.py 尚未調整。
