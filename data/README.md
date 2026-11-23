# 餐廳資料

| 檔案 | 內容 | 進 git |
|---|---|---|
| `restaurants.json` | 32 家（DL-001～DL-033，DL-011 併入 DL-010），依菜單照片人工整理 | 是 |
| `restaurants.draft.json` | 2026-09-07 原始草稿，保留不改寫 | 是 |
| `restaurants_google.json` | 90 家（DL-101～DL-190），Google Places 收集 | 否（Google 使用條款不得轉發） |
| `places_raw.json`、`places_review.csv` | Google 收集的原始資料與人工檢查表 | 否 |
| `embeddings/*.json` | Embedding 向量快取；含 Google 資料的快取不進 git | 部分 |

- [欄位說明](DATA_FIELDS.md)
- [資料預覽與限制](RESTAURANTS_PREVIEW.md)（第 3 項當時的預覽）

## 32 家菜單資料

- 每家都有主餐價格範圍、菜單摘要與完整主餐清單（menu_items，共 581 道）。
- 27 家有逐道價格（menu_prices，共 482 道，取最小份的單品價）；5 家套餐或鍋物組合店只有價格範圍。
- 每道菜由關鍵字表判斷分類；判斷不了或會判錯的 21 道另外人工標註（item_tags）。
- 3 家有營業時段、1 家有地標標記。
- 價格是菜單上的單品價，不是每人整餐費用；都沒有向店家確認即時營業或價格。

## 驗證

```powershell
$env:PYTHONPATH = 'src'
python -B -m food_crag.dataset
```

預期輸出 records=32、with_price=32、with_menu=32。檢查內容包括唯一 ID、店名與地址重複、必要欄位、斗六範圍、型別、日期、來源網址，以及菜名、價格、人工分類必須出自菜單清單。格式檢查不能代替實地查核。
