# Configuration MCPU v2 全量复裁台账

本台账覆盖全部 406 个 Configuration mention；规则制定与复裁未读取模型预测。

- 门禁：`gate_passed`
- 跨度修正及/或 CPE 修正：17 项
- 非通配符 CPE 版本：4 项

| 文档 | 实体 | 裁决 | 新跨度 | 新 CPE | 依据 |
|---|---|---|---|---|---|
| aa22-228a | E53 | REVISE_SPAN | `ZCS` | `cpe:2.3:a:zimbra:collaboration:*:*:*:*:*:*:*:*` | Remove the generic deployment/class suffix; ZCS is the independently identifiable affected product mention. |
| aa23-074a-telerik-cve-18935 | E81 | REVISE_SPAN | `Progress Telerik user interface (UI) for ASP.NET AJAX` | `cpe:2.3:a:progress:telerik_ui_for_asp.net_ajax:*:*:*:*:*:*:*:*` | Progress is an immediately adjacent vendor token in the same continuous product noun phrase. |
| aa23-215a | E165 | REVISE_SPAN | `Microsoft Exchange` | `cpe:2.3:a:microsoft:exchange_server:*:*:*:*:*:*:*:*` | Retain the adjacent vendor-product identity and remove the generic email-server class suffix. |
| aa23-215a | E168 | REVISE_SPAN | `Apache’s Log4j` | `cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*` | Retain the adjacent vendor-product identity and remove the generic library class suffix. |
| aa23-215a | E206 | REVISE_SPAN | `SMA 100 Series` | `cpe:2.3:o:sonicwall:sma_100_firmware:*:*:*:*:*:*:*:*` | SMA 100 Series is the product-family identity; Appliances is a generic class suffix in this occurrence. |
| aa23-215a | E224 | REVISE_SPAN | `SD-WAN WANOP` | `cpe:2.3:a:citrix:sd-wan:*:*:*:*:*:*:*:*` | Remove the generic appliance class suffix while retaining the named SD-WAN WANOP product form. |
| aa23-325a | E12 | REVISE_SPAN | `NetScaler Gateway` | `cpe:2.3:a:citrix:netscaler_gateway:*:*:*:*:*:*:*:*` | NetScaler Gateway is the formal product identity; appliances is a descriptive plural suffix. |
| aa23-325a | E20 | REVISE_SPAN | `Citrix NetScaler` | `cpe:2.3:a:citrix:netscaler_application_delivery_controller:*:*:*:*:*:*:*:*` | Retain the adjacent vendor-product identity and remove the explanatory product-class gloss. |
| aa23-325a-citrix-bleed-cve-4966 | E23 | REVISE_SPAN | `Citrix NetScaler` | `cpe:2.3:a:citrix:netscaler_application_delivery_controller:*:*:*:*:*:*:*:*` | Retain the adjacent vendor-product identity and remove the explanatory product-class gloss. |
| aa23-339a | E56 | REVISE_SPAN_AND_CPE | `ColdFusion` | `cpe:2.3:a:adobe:coldfusion:*:*:*:*:*:*:*:*` | 2016 is an ordinary ColdFusion release version and is excluded from the product surface. MCPU product mentions normalize to the ColdFusion family; ordinary release constraints remain in source evidence. |
| aa23-339a | E57 | REVISE_SPAN_AND_CPE | `ColdFusion` | `cpe:2.3:a:adobe:coldfusion:*:*:*:*:*:*:*:*` | 11 is an ordinary ColdFusion release version and is excluded from the product surface. MCPU product mentions normalize to the ColdFusion family; ordinary release constraints remain in source evidence. |
| aa23-339a-coldfusion-cve-26360 | E3 | REVISE_SPAN_AND_CPE | `Adobe ColdFusion` | `cpe:2.3:a:adobe:coldfusion:*:*:*:*:*:*:*:*` | Retain the adjacent vendor-product identity and exclude the ordinary release phrase versions 2018. MCPU product mentions normalize to the ColdFusion family; ordinary release constraints remain in source evidence. |
| aa23-339a-coldfusion-cve-26360 | E4 | REVISE_CPE | `Adobe ColdFusion` | `cpe:2.3:a:adobe:coldfusion:*:*:*:*:*:*:*:*` | The surface is a product-family mention; do not encode an ordinary runtime release as the entity identity. |
| aa23-339a-coldfusion-cve-26360 | E5 | REVISE_CPE | `Adobe ColdFusion` | `cpe:2.3:a:adobe:coldfusion:*:*:*:*:*:*:*:*` | The surface is a product-family mention; do not encode an ordinary runtime release as the entity identity. |
| aa23-339a-coldfusion-cve-26360 | E44 | REVISE_SPAN_AND_CPE | `ColdFusion` | `cpe:2.3:a:adobe:coldfusion:*:*:*:*:*:*:*:*` | 2016 is an ordinary ColdFusion release version and is excluded from the product surface. MCPU product mentions normalize to the ColdFusion family; ordinary release constraints remain in source evidence. |
| aa23-339a-coldfusion-cve-26360 | E45 | REVISE_SPAN_AND_CPE | `ColdFusion` | `cpe:2.3:a:adobe:coldfusion:*:*:*:*:*:*:*:*` | 11 is an ordinary ColdFusion release version and is excluded from the product surface. MCPU product mentions normalize to the ColdFusion family; ordinary release constraints remain in source evidence. |
| aa24-207a | E108 | REVISE_SPAN | `Apache’s Log4j` | `cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*` | Retain the adjacent vendor-product identity and remove the generic software-library class suffix. |
