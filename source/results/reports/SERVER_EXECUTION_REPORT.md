# Server execution report

All planned training records passed the run-level audit.

```json
{
  "utc": "2026-11-12T17:09:18.018654+00:00",
  "git_commit": "370e1407a80d4b49fdbed7f76af3b9ae06afc23b",
  "dirty_worktree": true,
  "phase_counts": {
    "part1_tuning": {
      "completed": 270
    },
    "part1_final": {
      "completed": 837
    },
    "bonus_tuning": {
      "completed": 36
    },
    "bonus_final": {
      "completed": 108
    }
  },
  "elapsed_workflow_sec": 59504.260390758514
}
```

Hardware: RTX 4070, 12282 MiB VRAM; driver535.230.02; i9-10900K (10 cores/20 threads), 31GiB RAM. All execution uses Conda ML; package exports and inventory are under source/results/server/. No concurrency is used. Detailed cleaning and split counts: preprocessing_report.md. Plans: source/results/plans/. Runtime pilot coverage and uncertainty: runtime_estimate.md/.json/.csv. Smoke/batch and VRAM measurements: smoke_matrix.csv. Status and missing/error rows: completeness_audit.csv. Monitor history: source/results/progress/history.csv. Stage logs: source/results/server/; job tracebacks: source/results/run_records/. No raw data or large model artifacts enter the results archive. Archive checksum: source/results/server/archive_manifest.json.

Authoritative CSVs: source/results/csv/run_records.csv, final_metrics_by_seed.csv, final_metrics_summary.csv, crossover_summary.csv, resource_summary.csv, bonus_metrics_by_seed.csv, bonus_metrics_summary.csv, bonus_label_efficiency.csv, bonus_representation_metrics.csv, complete_project_status.csv. These distinguish measured per-run outputs from derived summaries. Missing values remain missing. Three seeds do not establish statistical significance. Source-label savings count only new target-domain labels. No LaTeX or presentation figures are generated.

Resume: `conda run --no-capture-output -n ML python scripts/server_execution.py run` after fixing the recorded failure. Completed runs are skipped. Status: `./run_all.sh status`. Attach: `tmux attach -t korean-sentiment-all`.

Part 1 measured performance, best method at each size (descriptive selection by test mean):

| dataset | train_size | model | test_macro_f1_mean | test_macro_f1_std |
| --- | --- | --- | --- | --- |
| nsmc | 200 | tfidf_svm | 0.6062272795687217 | 0.009588162855026354 |
| nsmc | 500 | tfidf_svm | 0.6399005011254895 | 0.005766817495642275 |
| nsmc | 1000 | tfidf_svm | 0.6702372237981645 | 0.0027774916770764934 |
| nsmc | 2000 | tfidf_svm | 0.6964962325732582 | 0.0019410688441309678 |
| nsmc | 4000 | tfidf_svm | 0.7229808096050734 | 0.0028495606818590134 |
| nsmc | 8000 | tfidf_lr | 0.7497463488936965 | 0.0022818125166174596 |
| nsmc | 16000 | tfidf_svm | 0.7673863749393245 | 0.0023712228157470603 |
| nsmc | 32000 | transformer | 0.7894481996364128 | 0.0010486954636090948 |
| nsmc | 64000 | transformer | 0.8044503499285419 | 0.001131991890557536 |
| nsmc | 130797 | transformer | 0.8184593834760818 | 0.00019133268840561797 |
| shopping | 200 | tfidf_svm | 0.7192750954202044 | 0.009553495864557327 |
| shopping | 500 | tfidf_svm | 0.7672763028438148 | 0.01521408486199607 |
| shopping | 1000 | tfidf_svm | 0.7980932769235961 | 0.004391997323560244 |
| shopping | 2000 | tfidf_svm | 0.8185973128220381 | 0.003089489675819695 |
| shopping | 4000 | tfidf_svm | 0.8361929459373596 | 0.0019351131447277294 |
| shopping | 8000 | tfidf_svm | 0.8524882588533691 | 0.00021640477948770814 |
| shopping | 16000 | tfidf_svm | 0.8636637819041578 | 0.002007656618944491 |
| shopping | 32000 | tfidf_svm | 0.8710047522129064 | 0.00042136596403900004 |
| shopping | 64000 | transformer | 0.8792078263831206 | 0.003512326762992898 |
| shopping | 128000 | transformer | 0.8882381498067408 | 0.0011795549170074535 |
| shopping | 159918 | transformer | 0.8914432446001092 | 0.0010602130413972153 |
| steam | 200 | tfidf_svm | 0.6050320767645837 | 0.005972140392185504 |
| steam | 500 | tfidf_svm | 0.6242055419203142 | 0.0058002737964639815 |
| steam | 1000 | tfidf_svm | 0.6449355744725386 | 0.004610088001360928 |
| steam | 2000 | tfidf_svm | 0.673248787007947 | 0.004370077923654222 |
| steam | 4000 | tfidf_svm | 0.7028674274297243 | 0.00517585911491956 |
| steam | 8000 | tfidf_lr | 0.7211728260544924 | 0.0025612830304469003 |
| steam | 16000 | tfidf_lr | 0.7351549989502328 | 0.0017618368303187442 |
| steam | 32000 | tfidf_lr | 0.7481776601937365 | 0.0034700252444733026 |
| steam | 64000 | transformer | 0.7625838190957457 | 0.0035751384136273445 |
| steam | 79828 | lstm | 0.7667925024787868 | 0.004381636273872713 |

Derived crossover results:

| dataset | first_observed_crossover | first_sustained_crossover |
| --- | --- | --- |
| nsmc | 64000 | 64000 |
| shopping |  |  |
| steam |  |  |

Full-data performance:

| dataset | model | train_size | test_macro_f1_mean | test_macro_f1_std |
| --- | --- | --- | --- | --- |
| nsmc | avg_mlp | 130797 | 0.8136135975219613 | 0.0003203506772315431 |
| nsmc | gru | 130797 | 0.8179764757998219 | 0.00018335240729895063 |
| nsmc | lstm | 130797 | 0.8163558192322323 | 0.0036164572282326606 |
| nsmc | mamba | 130797 | 0.8071915343789882 | 0.0012476048602285737 |
| nsmc | tfidf_lr | 130797 | 0.8007402243662917 | 1.3597399555105182e-16 |
| nsmc | tfidf_rf | 130797 | 0.7538721982444109 | 0.00013560664595427081 |
| nsmc | tfidf_svm | 130797 | 0.8004275286718996 | 0.0 |
| nsmc | tfidf_xgb | 130797 | 0.6893642896020822 | 0.0003124361945558154 |
| nsmc | transformer | 130797 | 0.8184593834760818 | 0.00019133268840561797 |
| shopping | avg_mlp | 159918 | 0.8861661353487132 | 0.001922975299761228 |
| shopping | gru | 159918 | 0.8903781651502373 | 0.0020606952979642292 |
| shopping | lstm | 159918 | 0.8896995509652275 | 0.0013064269109906252 |
| shopping | mamba | 159918 | 0.8808979052102667 | 0.0008046466362885252 |
| shopping | tfidf_lr | 159918 | 0.8833722158552328 | 0.0 |
| shopping | tfidf_rf | 159918 | 0.8508823075608293 | 0.0002472945783509801 |
| shopping | tfidf_svm | 159918 | 0.8830156989207262 | 0.0 |
| shopping | tfidf_xgb | 159918 | 0.8036820871517275 | 0.0005219052690916891 |
| shopping | transformer | 159918 | 0.8914432446001092 | 0.0010602130413972153 |
| steam | avg_mlp | 79828 | 0.763888188380065 | 0.001374902603184577 |
| steam | gru | 79828 | 0.766032105614124 | 0.0042419529580466474 |
| steam | lstm | 79828 | 0.7667925024787868 | 0.004381636273872713 |
| steam | mamba | 79828 | 0.7539113820998216 | 0.001382360446379409 |
| steam | tfidf_lr | 79828 | 0.7582433080047869 | 1.3597399555105182e-16 |
| steam | tfidf_rf | 79828 | 0.724818441989183 | 0.0004065610870955997 |
| steam | tfidf_svm | 79828 | 0.7567100404996635 | 1.3597399555105182e-16 |
| steam | tfidf_xgb | 79828 | 0.682114771881175 | 0.0009086936990425325 |
| steam | transformer | 79828 | 0.7663853698263328 | 0.002140733066038489 |

Part 2 measured target-domain macro-F1 and source retention:

| target_domain | target_train_size | method | target_macro_f1_mean | target_macro_f1_std | mean_source_macro_f1_mean | source_retention_difference_vs_pooled_ce |
| --- | --- | --- | --- | --- | --- | --- |
| nsmc | 200 | best_tfidf | 0.6062272795687218 | 0.009588162855026354 |  |  |
| nsmc | 500 | best_tfidf | 0.6399005011254895 | 0.005766817495642275 |  |  |
| nsmc | 1000 | best_tfidf | 0.6702372237981645 | 0.0027774916770764934 |  |  |
| nsmc | 2000 | best_tfidf | 0.6964962325732582 | 0.0019410688441309678 |  |  |
| nsmc | 200 | db_cdsupcon | 0.6358878952446212 | 0.007488532094539523 | 0.6916908816117876 | 0.01030721292900083 |
| nsmc | 500 | db_cdsupcon | 0.6535948365796879 | 0.003387567755193938 | 0.7004966160732105 | 0.007612099979755427 |
| nsmc | 1000 | db_cdsupcon | 0.6652684488979516 | 0.007411075149247406 | 0.7109251663186117 | 0.005491725906667355 |
| nsmc | 2000 | db_cdsupcon | 0.6878026437390939 | 0.0062193500042392845 | 0.7173141198072005 | -0.004317184533785157 |
| nsmc | 200 | pooled_ce | 0.6281829502161139 | 0.008272588444884146 | 0.6813836686827868 | 0.0 |
| nsmc | 500 | pooled_ce | 0.6473468935266214 | 0.01329787507062706 | 0.692884516093455 | 0.0 |
| nsmc | 1000 | pooled_ce | 0.6654720863150606 | 0.010671783747777516 | 0.7054334404119443 | 0.0 |
| nsmc | 2000 | pooled_ce | 0.6871288030042538 | 0.0037837101007680172 | 0.7216313043409857 | 0.0 |
| nsmc | 200 | supcon | 0.6307593141490804 | 0.007915573288767243 | 0.6842919219608747 | 0.002908253278087969 |
| nsmc | 500 | supcon | 0.648273688684775 | 0.013220015514973538 | 0.6919367443029807 | -0.0009477717904743388 |
| nsmc | 1000 | supcon | 0.6601934074125801 | 0.017992870896288374 | 0.7057035790575273 | 0.0002701386455830068 |
| nsmc | 2000 | supcon | 0.6852202948868372 | 0.00678459972479278 | 0.7191291803993828 | -0.002502123941602874 |
| nsmc | 200 | target_transformer | 0.5076552117136995 | 0.035188721836243234 |  |  |
| nsmc | 500 | target_transformer | 0.5125525927361129 | 0.0612850867557451 |  |  |
| nsmc | 1000 | target_transformer | 0.5472389251442422 | 0.05970304057578996 |  |  |
| nsmc | 2000 | target_transformer | 0.5790424089097002 | 0.011726668110403047 |  |  |
| shopping | 200 | best_tfidf | 0.7198683307398891 | 0.008595163679213383 |  |  |
| shopping | 500 | best_tfidf | 0.7672763028438148 | 0.01521408486199607 |  |  |
| shopping | 1000 | best_tfidf | 0.7980932769235961 | 0.004391997323560244 |  |  |
| shopping | 2000 | best_tfidf | 0.8185973128220381 | 0.003089489675819695 |  |  |
| shopping | 200 | db_cdsupcon | 0.697251262913204 | 0.03768899227320333 | 0.6231538538729945 | 0.016430311878796844 |
| shopping | 500 | db_cdsupcon | 0.7429513992017395 | 0.004583035204940213 | 0.6404503733720408 | 0.00371527419602391 |
| shopping | 1000 | db_cdsupcon | 0.7534569520649305 | 0.016959371259020632 | 0.643026366966768 | -0.0011634502872364116 |
| shopping | 2000 | db_cdsupcon | 0.7788082384253409 | 0.006939496756578327 | 0.6659439176405334 | 0.03953340906094438 |
| shopping | 200 | pooled_ce | 0.6911358992819839 | 0.03289827800236422 | 0.6067235419941976 | 0.0 |
| shopping | 500 | pooled_ce | 0.7402127375371198 | 0.006681902195365841 | 0.6367350991760169 | 0.0 |
| shopping | 1000 | pooled_ce | 0.7586073281397253 | 0.015765737413917447 | 0.6441898172540044 | 0.0 |
| shopping | 2000 | pooled_ce | 0.754475513969402 | 0.021871298192701664 | 0.626410508579589 | 0.0 |
| shopping | 200 | supcon | 0.6948455423381864 | 0.03631135895533314 | 0.6167816472699125 | 0.010058105275714824 |
| shopping | 500 | supcon | 0.7421529467226106 | 0.00854648079933698 | 0.6408381869042729 | 0.004103087728255961 |
| shopping | 1000 | supcon | 0.7551547667072297 | 0.017049734775056787 | 0.6474966205410099 | 0.0033068032870054997 |
| shopping | 2000 | supcon | 0.7803812200730399 | 0.007020549188784037 | 0.667446091261996 | 0.041035582682407035 |
| shopping | 200 | target_transformer | 0.6392007804356907 | 0.027361737420254315 |  |  |
| shopping | 500 | target_transformer | 0.647035716581782 | 0.055339889995912245 |  |  |
| shopping | 1000 | target_transformer | 0.7250599007469841 | 0.02314163031522445 |  |  |
| shopping | 2000 | target_transformer | 0.7175107308187548 | 0.07659682825501979 |  |  |
| steam | 200 | best_tfidf | 0.6051978018487142 | 0.00578070037952211 |  |  |
| steam | 500 | best_tfidf | 0.6242681884174036 | 0.0056922420283337894 |  |  |
| steam | 1000 | best_tfidf | 0.6450034394037018 | 0.004496300456485446 |  |  |
| steam | 2000 | best_tfidf | 0.673374350124369 | 0.004372381865589323 |  |  |
| steam | 200 | db_cdsupcon | 0.6278178992822822 | 0.014993000707213131 | 0.7204592060368538 | 0.006625833713980289 |
| steam | 500 | db_cdsupcon | 0.6283110669680864 | 0.009178797155278624 | 0.697686779599579 | -0.007595172874031109 |
| steam | 1000 | db_cdsupcon | 0.6497209366135032 | 0.008428014649232332 | 0.7253475938714066 | -0.0007791363404021645 |
| steam | 2000 | db_cdsupcon | 0.661434695074631 | 0.006568934371953318 | 0.728088427212926 | -0.0011333348206751248 |
| steam | 200 | pooled_ce | 0.6271047443803827 | 0.010395944580296981 | 0.7138333723228735 | 0.0 |
| steam | 500 | pooled_ce | 0.628318626531326 | 0.013342297726027694 | 0.7052819524736101 | 0.0 |
| steam | 1000 | pooled_ce | 0.6517873426981057 | 0.0034726585228871615 | 0.7261267302118087 | 0.0 |
| steam | 2000 | pooled_ce | 0.6628847549806008 | 0.004115629858805105 | 0.7292217620336011 | 0.0 |
| steam | 200 | supcon | 0.6170123489414818 | 0.0019671449607850445 | 0.7119249956956399 | -0.0019083766272335323 |
| steam | 500 | supcon | 0.6392919824523123 | 0.0013631124918545396 | 0.7199185740216573 | 0.014636621548047124 |
| steam | 1000 | supcon | 0.6466902872827399 | 0.005744197859127729 | 0.7218880779540715 | -0.00423865225773723 |
| steam | 2000 | supcon | 0.665065617539478 | 0.009711540463967308 | 0.7184426971986948 | -0.010779064834906316 |
| steam | 200 | target_transformer | 0.4917537252432503 | 0.026021242234684917 |  |  |
| steam | 500 | target_transformer | 0.5420068870374372 | 0.011383313587132625 |  |  |
| steam | 1000 | target_transformer | 0.580158717773092 | 0.018704428757064052 |  |  |
| steam | 2000 | target_transformer | 0.5733099497813458 | 0.05484676371711659 |  |  |

Derived target-label efficiency (unobserved thresholds remain missing):

| target_domain | method | crossover_threshold_size | target_label_crossover_shift | target_training_labels_saved |
| --- | --- | --- | --- | --- |
| nsmc | best_tfidf |  |  |  |
| nsmc | db_cdsupcon | 200 |  |  |
| nsmc | pooled_ce | 200 |  |  |
| nsmc | supcon | 200 |  |  |
| nsmc | target_transformer |  |  |  |
| shopping | best_tfidf |  |  |  |
| shopping | db_cdsupcon |  |  |  |
| shopping | pooled_ce |  |  |  |
| shopping | supcon |  |  |  |
| shopping | target_transformer |  |  |  |
| steam | best_tfidf |  |  |  |
| steam | db_cdsupcon | 200 |  |  |
| steam | pooled_ce | 200 |  |  |
| steam | supcon | 200 |  |  |
| steam | target_transformer |  |  |  |

Measured training/inference cost and cosine-separation diagnostics are grouped by phase, dataset, model and objective in server/cost_and_representation_summary.csv; these are means across sizes and seeds, so use run-level resources for matched comparisons. Projection heads are excluded from deployed inference. Measured child-process CPU time for recorded production phases and completed GPU-assigned job wall time are reported below. GPU utilization-hours and the instantaneous storage peak remain unavailable; discrete storage samples provide the observed peak.

Interpretation boundaries: noisy rating/recommendation labels, three specific domains, from-scratch representations, validation-selected configurations, finite tested label budgets and three seeds. Crossovers summarize observed means and do not demonstrate significance.

Full-data costs (means of three seed measurements; latency columns average per-run medians, memory columns take the largest observed seed peak):

| dataset | model | train_size | device | fit_time_sec_mean | classifier_latency_ms_p50_mean | end_to_end_latency_ms_p50_mean | serialized_size_mb_mean | peak_host_ram_mb_max | peak_gpu_memory_mb_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nsmc | avg_mlp | 130797 | cuda | 45.072357390658 | 0.5327548333333333 | 0.5982955 | 10.60519822438558 | 3012.6328125 | 137.8642578125 |
| nsmc | gru | 130797 | cuda | 134.60476432734868 | 0.8774458333333334 | 0.9443060000000001 | 12.036060333251953 | 3301.0703125 | 183.271484375 |
| nsmc | lstm | 130797 | cuda | 152.36953558433257 | 0.8833273333333334 | 0.9451755 | 12.13156000773112 | 3302.078125 | 175.56103515625 |
| nsmc | mamba | 130797 | cuda | 106.43070793367224 | 1.8601351666666666 | 1.9293488333333333 | 11.04233169555664 | 3448.9453125 | 351.7890625 |
| nsmc | tfidf_lr | 130797 | cpu | 5.077946629006571 | 0.156028 | 0.5016965 | 0.41855430603027344 | 1702.296875 | None |
| nsmc | tfidf_rf | 130797 | cpu | 105.88254770364922 | 74.0720995 | 74.18312466666667 | 641.3751122156779 | 2729.84375 | None |
| nsmc | tfidf_svm | 130797 | cpu | 5.001292776995494 | 0.153742 | 0.49201333333333336 | 0.41838645935058594 | 1691.453125 | None |
| nsmc | tfidf_xgb | 130797 | cpu | 34.29127239567848 | 0.299507 | 0.6670203333333333 | 0.8880313237508138 | 2528.33984375 | None |
| nsmc | transformer | 130797 | cuda | 235.89552910769513 | 1.8102515 | 1.8716993333333332 | 11.226367314656576 | 3353.4375 | 334.69189453125 |
| shopping | avg_mlp | 159918 | cuda | 67.0098017626636 | 0.5521025 | 0.6218566666666666 | 10.618341128031412 | 4345.6796875 | 140.181640625 |
| shopping | gru | 159918 | cuda | 160.16004930935256 | 0.9778483333333332 | 1.0448948333333332 | 12.049121856689453 | 4335.3203125 | 183.1484375 |
| shopping | lstm | 159918 | cuda | 159.80137913534418 | 1.4690028333333334 | 1.5821956666666666 | 8.286502838134766 | 4345.96875 | 150.10693359375 |
| shopping | mamba | 159918 | cuda | 132.23415925200484 | 1.9048253333333334 | 1.9847089999999998 | 11.05539321899414 | 2399.203125 | 348.91162109375 |
| shopping | tfidf_lr | 159918 | cpu | 5.944887114669352 | 0.153556 | 0.5035833333333333 | 0.43059349060058594 | 3483.09765625 | None |
| shopping | tfidf_rf | 159918 | cpu | 125.94310708666065 | 73.15884466666667 | 72.16924266666668 | 766.3974765141805 | 4225.1484375 | None |
| shopping | tfidf_svm | 159918 | cpu | 6.004345720653267 | 0.195522 | 0.5219346666666667 | 0.43044090270996094 | 3489.2890625 | None |
| shopping | tfidf_xgb | 159918 | cpu | 44.291077859001234 | 0.3187778333333333 | 0.6929953333333333 | 0.9264955520629883 | 4286.75 | None |
| shopping | transformer | 159918 | cuda | 289.7109227880137 | 1.8876148333333334 | 1.9550883333333333 | 11.239408493041992 | 2291.9921875 | 332.84619140625 |
| steam | avg_mlp | 79828 | cuda | 28.058506697358098 | 0.5483995 | 0.6176328333333334 | 10.604262351989746 | 2876.35546875 | 138.294921875 |
| steam | gru | 79828 | cuda | 88.91311532301673 | 0.9625213333333332 | 1.0254113333333335 | 12.035144805908203 | 3115.53515625 | 182.75048828125 |
| steam | lstm | 79828 | cuda | 95.08196006465005 | 0.9551261666666666 | 1.0266515 | 12.130563100179037 | 3120.2109375 | 175.77783203125 |
| steam | mamba | 79828 | cuda | 66.21681312733563 | 1.9076163333333334 | 1.9854956666666668 | 11.03940200805664 | 3137.48828125 | 350.01416015625 |
| steam | tfidf_lr | 79828 | cpu | 3.431958887667861 | 0.15798233333333334 | 0.5173446666666667 | 0.41844749450683594 | 2162.45703125 | None |
| steam | tfidf_rf | 79828 | cpu | 62.84437842434272 | 64.21727633333334 | 67.2856185 | 473.32436656951904 | 2861.81640625 | None |
| steam | tfidf_svm | 79828 | cpu | 3.45937872332676 | 0.15402000000000002 | 0.5147198333333334 | 0.41829490661621094 | 2162.50390625 | None |
| steam | tfidf_xgb | 79828 | cpu | 28.28885230666492 | 0.31397933333333333 | 0.7096396666666666 | 0.8941615422566732 | 2871.328125 | None |
| steam | transformer | 79828 | cuda | 105.46502465667436 | 1.8957861666666667 | 1.9494258333333334 | 11.225248336791992 | 3126.890625 | 461.95654296875 |

Part 2 matched training and inference costs. Fit-time ratios include data preparation and early stopping, rather than measuring only the extra contrastive-loss operation. Serialized sizes are training artefacts; projection heads are bypassed for deployed inference. Device differences matter when comparing CPU TF-IDF and CUDA neural latency.

| target_domain | target_train_size | method | device | fit_time_sec_mean | fit_time_ratio_vs_pooled_ce | classifier_latency_ms_p50_mean | end_to_end_latency_ms_p50_mean | deployed_parameter_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nsmc | 200 | best_tfidf | cpu | 0.6616782439911427 | 0.037267248262078 | 0.15801333333333334 | 0.5035381666666666 | None |
| nsmc | 200 | db_cdsupcon | cuda | 24.364952256340377 | 1.372290434023805 | 1.8665041666666669 | 1.9362768333333333 | 2841858.0 |
| nsmc | 200 | pooled_ce | cuda | 17.754953071339212 | 1.0 | 1.8969209999999999 | 1.950698 | 2841858.0 |
| nsmc | 200 | supcon | cuda | 20.706122804665938 | 1.1662167014167233 | 1.8926993333333335 | 1.9579065 | 2841858.0 |
| nsmc | 200 | target_transformer | cuda | 3.6687791783285015 | 0.20663412421240349 | 1.827483 | 1.8920245 | 453378.0 |
| nsmc | 500 | best_tfidf | cpu | 0.6638208776712418 | 0.033062062778687944 | 0.15558400000000003 | 0.49643333333333334 | None |
| nsmc | 500 | db_cdsupcon | cuda | 26.621190363308415 | 1.3258869924711851 | 1.8642126666666667 | 1.930601 | 2841858.0 |
| nsmc | 500 | pooled_ce | cuda | 20.07802362831232 | 1.0 | 1.8639268333333334 | 1.9380525000000002 | 2841858.0 |
| nsmc | 500 | supcon | cuda | 24.01040932831044 | 1.19585521826227 | 1.8914805000000001 | 1.95124 | 2841858.0 |
| nsmc | 500 | target_transformer | cuda | 4.312401244009379 | 0.21478215803712863 | 1.8938905000000001 | 1.9657326666666668 | 673452.6666666666 |
| nsmc | 1000 | best_tfidf | cpu | 0.6835436446514601 | 0.029631120998808307 | 0.1567086666666667 | 0.4948725 | None |
| nsmc | 1000 | db_cdsupcon | cuda | 27.821917009326473 | 1.2060599140564252 | 1.8640438333333333 | 1.9172020000000003 | 2841858.0 |
| nsmc | 1000 | pooled_ce | cuda | 23.068436886979423 | 1.0 | 1.9136695 | 1.9819375000000001 | 2841858.0 |
| nsmc | 1000 | supcon | cuda | 26.547468697982065 | 1.1508135045320873 | 1.8963528333333333 | 1.9658636666666667 | 2841858.0 |
| nsmc | 1000 | target_transformer | cuda | 7.574527946650051 | 0.32835029021517104 | 1.8717765 | 1.9439913333333336 | 988674.0 |
| nsmc | 2000 | best_tfidf | cpu | 0.7004660683257194 | 0.02378059855940478 | 0.15311566666666668 | 0.49138583333333336 | None |
| nsmc | 2000 | db_cdsupcon | cuda | 34.927525759655204 | 1.1857783072190142 | 1.8443873333333334 | 1.8906955000000003 | 2841858.0 |
| nsmc | 2000 | pooled_ce | cuda | 29.455359022016637 | 1.0 | 1.8419733333333335 | 1.925716666666667 | 2841858.0 |
| nsmc | 2000 | supcon | cuda | 33.81920956529211 | 1.1481513275738273 | 1.8907788333333333 | 1.9489513333333335 | 2841858.0 |
| nsmc | 2000 | target_transformer | cuda | 5.9269785033344915 | 0.20121902092262142 | 1.7875915000000002 | 1.8502586666666667 | 1554135.3333333333 |
| shopping | 200 | best_tfidf | cpu | 0.9315159746717351 | 0.06743444123577609 | 21.511243833333335 | 21.830690166666667 | None |
| shopping | 200 | db_cdsupcon | cuda | 20.968344196327962 | 1.5179434523568875 | 1.9049969999999998 | 1.9708994999999998 | 2841858.0 |
| shopping | 200 | pooled_ce | cuda | 13.813653047332386 | 1.0 | 1.8416813333333335 | 1.9068581666666666 | 2841858.0 |
| shopping | 200 | supcon | cuda | 17.637070707976818 | 1.276785412775572 | 1.8818526666666668 | 1.953887 | 2841858.0 |
| shopping | 200 | target_transformer | cuda | 8.570801080340365 | 0.6204586904689567 | 1.8566840000000002 | 1.9162931666666667 | 468994.0 |
| shopping | 500 | best_tfidf | cpu | 0.7382765256528122 | 0.04180406577897658 | 0.15406733333333333 | 0.5035845 | None |
| shopping | 500 | db_cdsupcon | cuda | 22.864608808323585 | 1.29468238176548 | 1.8842288333333332 | 1.9715731666666667 | 2841858.0 |
| shopping | 500 | pooled_ce | cuda | 17.66040005668765 | 1.0 | 1.9081578333333333 | 1.9844018333333333 | 2841858.0 |
| shopping | 500 | supcon | cuda | 23.005261980327003 | 1.3026467071234527 | 1.9238601666666666 | 2.0003843333333333 | 2841858.0 |
| shopping | 500 | target_transformer | cuda | 7.81489567332513 | 0.4425095495141845 | 1.83993 | 1.8949745 | 682498.0 |
| shopping | 1000 | best_tfidf | cpu | 0.7605279370133454 | 0.03611775322842016 | 0.15248016666666667 | 0.4996403333333333 | None |
| shopping | 1000 | db_cdsupcon | cuda | 24.148407587005448 | 1.146816814530072 | 1.8950466666666665 | 1.9600035 | 2841858.0 |
| shopping | 1000 | pooled_ce | cuda | 21.056900527658097 | 1.0 | 1.8866163333333332 | 1.9556148333333334 | 2841858.0 |
| shopping | 1000 | supcon | cuda | 23.92364861734677 | 1.136142927869333 | 1.8759071666666667 | 1.9454956666666667 | 2841858.0 |
| shopping | 1000 | target_transformer | cuda | 10.29357985399353 | 0.488845917302643 | 1.8314281666666667 | 1.9140986666666666 | 1001900.6666666666 |
| shopping | 2000 | best_tfidf | cpu | 0.7659342239882486 | 0.03914820918152692 | 0.15394699999999997 | 0.49645900000000004 | None |
| shopping | 2000 | db_cdsupcon | cuda | 32.671785689987395 | 1.6699108885168032 | 1.9009896666666666 | 1.9635904999999998 | 2841858.0 |
| shopping | 2000 | pooled_ce | cuda | 19.56498751799033 | 1.0 | 1.8894073333333334 | 1.9636446666666665 | 2841858.0 |
| shopping | 2000 | supcon | cuda | 33.796109244343825 | 1.7273769898022038 | 1.9243926666666666 | 1.9925066666666666 | 2841858.0 |
| shopping | 2000 | target_transformer | cuda | 11.77175552432891 | 0.6016745736998087 | 1.8070843333333333 | 1.8674353333333336 | 1563180.6666666667 |
| steam | 200 | best_tfidf | cpu | 0.36675646068761125 | 0.01914994650348765 | 0.15541783333333334 | 0.5247145 | None |
| steam | 200 | db_cdsupcon | cuda | 21.927722841326613 | 1.1449415739464661 | 1.9006958333333335 | 1.9596828333333332 | 2836226.0 |
| steam | 200 | pooled_ce | cuda | 19.151826905668713 | 1.0 | 1.8765833333333333 | 1.9051648333333333 | 2836226.0 |
| steam | 200 | supcon | cuda | 17.950404066340223 | 0.9372684994885326 | 1.8507191666666667 | 1.8972638333333334 | 2836226.0 |
| steam | 200 | target_transformer | cuda | 2.446406496669321 | 0.12773750038150217 | 1.904269 | 1.9534931666666668 | 506156.6666666667 |
| steam | 500 | best_tfidf | cpu | 0.36521731099734706 | 0.02546486023054462 | 0.1576805 | 0.5133381666666667 | None |
| steam | 500 | db_cdsupcon | cuda | 15.092348568005642 | 1.0523174429640212 | 1.9391811666666667 | 2.012579833333333 | 2841858.0 |
| steam | 500 | pooled_ce | cuda | 14.342011214311546 | 1.0 | 1.8991123333333333 | 1.9634853333333335 | 2841858.0 |
| steam | 500 | supcon | cuda | 21.37028060933032 | 1.4900476850837652 | 1.8763221666666665 | 1.9460104999999999 | 2841858.0 |
| steam | 500 | target_transformer | cuda | 3.3966547320208824 | 0.23683252517829878 | 1.9022283333333334 | 1.9847565000000003 | 778114.0 |
| steam | 1000 | best_tfidf | cpu | 0.3698119413262854 | 0.016800631964878965 | 0.1583125 | 0.5241705 | None |
| steam | 1000 | db_cdsupcon | cuda | 26.34073620266281 | 1.1966650212477778 | 1.838593666666667 | 1.9070749999999999 | 2841858.0 |
| steam | 1000 | pooled_ce | cuda | 22.011787538668916 | 1.0 | 1.8609056666666666 | 1.9358588333333333 | 2841858.0 |
| steam | 1000 | supcon | cuda | 23.803931061333667 | 1.0814174459714563 | 1.8710353333333334 | 1.9309255 | 2841858.0 |
| steam | 1000 | target_transformer | cuda | 5.474546241341159 | 0.2487097529777544 | 1.8908921666666665 | 1.9718278333333332 | 1173719.3333333333 |
| steam | 2000 | best_tfidf | cpu | 0.39704925200203434 | 0.016612095277514432 | 0.1577295 | 0.5148891666666667 | None |
| steam | 2000 | db_cdsupcon | cuda | 22.409337743689928 | 0.9375815514753068 | 1.8883558333333335 | 1.9449755 | 2841858.0 |
| steam | 2000 | pooled_ce | cuda | 23.901214468680944 | 1.0 | 1.8693968333333333 | 1.9354820000000001 | 2841858.0 |
| steam | 2000 | supcon | cuda | 24.95790523798981 | 1.044210756348532 | 1.894194 | 1.9651306666666668 | 2841858.0 |
| steam | 2000 | target_transformer | cuda | 7.12305246668014 | 0.2980205242714282 | 1.8649946666666668 | 1.9295901666666666 | 1846274.0 |

Representation cosine diagnostics, averaged over the three seeds using the fixed evaluation sample and pair seed:

| target_domain | target_train_size | method | same_sentiment_cross_domain_cosine_mean | different_sentiment_cross_domain_cosine_mean | cross_domain_sentiment_separation_gap_mean | same_domain_same_sentiment_cosine_mean |
| --- | --- | --- | --- | --- | --- | --- |
| nsmc | 200 | db_cdsupcon | 0.7198563267822271 | 0.6649178116495174 | 0.054938515132709864 | 0.7229708736852958 |
| nsmc | 200 | pooled_ce | 0.6377729114886002 | 0.5872295835628516 | 0.05054332792574856 | 0.641037204995856 |
| nsmc | 200 | supcon | 0.6608573450695387 | 0.6114434822336584 | 0.049413862835880305 | 0.6611011214483901 |
| nsmc | 500 | db_cdsupcon | 0.6632419848622008 | 0.5802136080279384 | 0.08302837683426235 | 0.6653067832111176 |
| nsmc | 500 | pooled_ce | 0.532924011914706 | 0.4401527121889677 | 0.09277129972573829 | 0.5497608060713453 |
| nsmc | 500 | supcon | 0.5825484053099723 | 0.49127164090379233 | 0.09127676440617999 | 0.5968564058910029 |
| nsmc | 1000 | db_cdsupcon | 0.6427692107276654 | 0.5575322654231862 | 0.08523694530447916 | 0.6561618529882659 |
| nsmc | 1000 | pooled_ce | 0.49774771387881334 | 0.39048408193972683 | 0.1072636319390865 | 0.5083340406680322 |
| nsmc | 1000 | supcon | 0.5695152460376046 | 0.4546264876056161 | 0.1148887584319886 | 0.5777354688423796 |
| nsmc | 2000 | db_cdsupcon | 0.5812176086184903 | 0.4513578146659973 | 0.12985979395249303 | 0.5903868776182702 |
| nsmc | 2000 | pooled_ce | 0.4927008609801817 | 0.3618663603217767 | 0.130834500658405 | 0.501333720571715 |
| nsmc | 2000 | supcon | 0.5498480851226136 | 0.4289102711534345 | 0.12093781396917913 | 0.5628332193328206 |
| shopping | 200 | db_cdsupcon | 0.6570901632146284 | 0.592984216993555 | 0.06410594622107345 | 0.6657463183333273 |
| shopping | 200 | pooled_ce | 0.65428366301741 | 0.6067610211348604 | 0.04752264188254962 | 0.6667511625308414 |
| shopping | 200 | supcon | 0.7043911749119086 | 0.645442681262831 | 0.058948493649077706 | 0.706511399223427 |
| shopping | 500 | db_cdsupcon | 0.6393452417688724 | 0.5456379428339662 | 0.09370729893490622 | 0.6515463624123551 |
| shopping | 500 | pooled_ce | 0.5519197411608449 | 0.4573767264660115 | 0.09454301469483332 | 0.562545952338685 |
| shopping | 500 | supcon | 0.6676316093429474 | 0.5781997852633549 | 0.08943182407959245 | 0.6797992645433065 |
| shopping | 1000 | db_cdsupcon | 0.6992867567782003 | 0.630078202720452 | 0.06920855405774823 | 0.7170513645983495 |
| shopping | 1000 | pooled_ce | 0.5906840418919583 | 0.5085634774349285 | 0.08212056445702993 | 0.6146915483171608 |
| shopping | 1000 | supcon | 0.7125250445479706 | 0.6431351228700487 | 0.0693899216779219 | 0.7312359513001846 |
| shopping | 2000 | db_cdsupcon | 0.6661960822186753 | 0.5484896598498624 | 0.1177064223688129 | 0.6852532719026402 |
| shopping | 2000 | pooled_ce | 0.6412680428222087 | 0.5530196741571531 | 0.0882483686650556 | 0.6624793181775173 |
| shopping | 2000 | supcon | 0.619048644148263 | 0.47176668216582623 | 0.14728196198243676 | 0.634111158686486 |
| steam | 200 | db_cdsupcon | 0.3940337483292014 | 0.24940312500374082 | 0.14463062332546064 | 0.41618778076187274 |
| steam | 200 | pooled_ce | 0.42717650647585853 | 0.280038897502616 | 0.1471376089732425 | 0.4441187405911354 |
| steam | 200 | supcon | 0.4213545136643852 | 0.2962979345265445 | 0.12505657913784066 | 0.45263366610150285 |
| steam | 500 | db_cdsupcon | 0.5105663630156 | 0.4171846820045082 | 0.09338168101109172 | 0.5259677900933085 |
| steam | 500 | pooled_ce | 0.43884003573253266 | 0.334638146264633 | 0.10420188946789972 | 0.46701174457024536 |
| steam | 500 | supcon | 0.4959112725941434 | 0.35765468260298666 | 0.13825658999115678 | 0.5079606662760116 |
| steam | 1000 | db_cdsupcon | 0.5407275598313965 | 0.3749378302661888 | 0.16578972956520766 | 0.5383830883719245 |
| steam | 1000 | pooled_ce | 0.408861913311112 | 0.23136178208601285 | 0.17750013122509908 | 0.41410214195434797 |
| steam | 1000 | supcon | 0.559426219779553 | 0.417792766516788 | 0.14163345326276508 | 0.5693598488547738 |
| steam | 2000 | db_cdsupcon | 0.5193180990650426 | 0.3362326732321632 | 0.18308542583287937 | 0.5295229226462278 |
| steam | 2000 | pooled_ce | 0.46233546178298757 | 0.27452631999911853 | 0.18780914178386907 | 0.4669029897217019 |
| steam | 2000 | supcon | 0.5849051008226236 | 0.40647008699719556 | 0.17843501382542803 | 0.5865897743152227 |

For complete curves, blank crossover thresholds mean the corresponding criterion was not observed within the tested sizes. They are not zero-label thresholds.

Final dataset splits and actual training-size grids:

```json
[
  {
    "dataset": "nsmc",
    "train_rows": "130797",
    "validation_rows": "14534",
    "test_rows": "49110",
    "n_full": "130797",
    "training_sizes": "[200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 130797]"
  },
  {
    "dataset": "shopping",
    "train_rows": "159918",
    "validation_rows": "19990",
    "test_rows": "19990",
    "n_full": "159918",
    "training_sizes": "[200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 159918]"
  },
  {
    "dataset": "steam",
    "train_rows": "79828",
    "validation_rows": "9979",
    "test_rows": "9979",
    "n_full": "79828",
    "training_sizes": "[200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 79828]"
  }
]
```

Cleaning removals, class balance and manifests (authoritative preprocessing report):


```json
{
  "statistics": [
    {
      "dataset": "nsmc",
      "train_rows": "130797",
      "validation_rows": "14534",
      "test_rows": "49110",
      "n_full": "130797",
      "training_sizes": "[200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 130797]"
    },
    {
      "dataset": "shopping",
      "train_rows": "159918",
      "validation_rows": "19990",
      "test_rows": "19990",
      "n_full": "159918",
      "training_sizes": "[200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 159918]"
    },
    {
      "dataset": "steam",
      "train_rows": "79828",
      "validation_rows": "9979",
      "test_rows": "9979",
      "n_full": "79828",
      "training_sizes": "[200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 79828]"
    }
  ],
  "cleaning": {
    "steam_cleaning_report": {
      "dataset": "steam",
      "n_full": 79828,
      "schedule": [
        200,
        500,
        1000,
        2000,
        4000,
        8000,
        16000,
        32000,
        64000,
        79828
      ],
      "sources": {
        "all": {
          "clean_rows": 99786,
          "conflicting_keys_removed": 107,
          "conflicting_occurrences_removed": 214,
          "duplicate_pairs_removed": 0,
          "empty_removed": 0,
          "malformed_rows": 0,
          "negative_rows": 49897,
          "neutral_rating_rows": 0,
          "parsed_rows": 100000,
          "positive_rows": 49889,
          "raw_rows": 100000
        }
      },
      "splits": {
        "test_rows": 9979,
        "train_rows": 79828,
        "validation_rows": 9979
      }
    },
    "nsmc_cleaning_report": {
      "dataset": "nsmc",
      "n_full": 130797,
      "schedule": [
        200,
        500,
        1000,
        2000,
        4000,
        8000,
        16000,
        32000,
        64000,
        130797
      ],
      "sources": {
        "test": {
          "clean_rows": 49110,
          "conflicting_keys_removed": 47,
          "conflicting_occurrences_removed": 306,
          "duplicate_pairs_removed": 581,
          "empty_removed": 3,
          "malformed_rows": 0,
          "negative_rows": 24423,
          "neutral_rating_rows": 0,
          "parsed_rows": 50000,
          "positive_rows": 24687,
          "raw_rows": 50000
        },
        "train": {
          "clean_rows": 146025,
          "conflicting_keys_removed": 157,
          "conflicting_occurrences_removed": 1451,
          "duplicate_pairs_removed": 2519,
          "empty_removed": 5,
          "malformed_rows": 0,
          "negative_rows": 73272,
          "neutral_rating_rows": 0,
          "parsed_rows": 150000,
          "positive_rows": 72753,
          "raw_rows": 150000
        }
      },
      "splits": {
        "official_cross_partition_train_rows_removed": 694,
        "test_rows": 49110,
        "train_rows": 130797,
        "validation_rows": 14534
      }
    },
    "shopping_cleaning_report": {
      "dataset": "shopping",
      "n_full": 159918,
      "schedule": [
        200,
        500,
        1000,
        2000,
        4000,
        8000,
        16000,
        32000,
        64000,
        128000,
        159918
      ],
      "sources": {
        "all": {
          "clean_rows": 199898,
          "conflicting_keys_removed": 10,
          "conflicting_occurrences_removed": 20,
          "duplicate_pairs_removed": 82,
          "empty_removed": 0,
          "malformed_rows": 0,
          "negative_rows": 99950,
          "neutral_rating_rows": 0,
          "parsed_rows": 200000,
          "positive_rows": 99948,
          "raw_rows": 200000
        }
      },
      "splits": {
        "test_rows": 19990,
        "train_rows": 159918,
        "validation_rows": 19990
      }
    }
  },
  "split_class_balance": {
    "nsmc": {
      "train": {
        "0": 65615,
        "1": 65182
      },
      "validation": {
        "0": 7291,
        "1": 7243
      },
      "test": {
        "0": 24423,
        "1": 24687
      }
    },
    "shopping": {
      "train": {
        "0": 79960,
        "1": 79958
      },
      "validation": {
        "0": 9995,
        "1": 9995
      },
      "test": {
        "0": 9995,
        "1": 9995
      }
    },
    "steam": {
      "train": {
        "0": 39917,
        "1": 39911
      },
      "validation": {
        "0": 4990,
        "1": 4989
      },
      "test": {
        "0": 4990,
        "1": 4989
      }
    }
  }
}
```

Checksums: source/data/manifests/download_manifest.csv. Splits: source/data/splits/. Nested orders and data hash: source/data/manifests/. Bonus uses target prefixes 200/500/1000/2000 and source prefixes 2000 from each other domain.

Numerical precision and timing boundaries:


Runs fix Python/NumPy/PyTorch seeds and set CUBLAS_WORKSPACE_CONFIG=:4096:8. PyTorch deterministic-algorithm mode uses warn_only=True. The repository pilot logged a warning that memory-efficient attention defaults to a non-deterministic algorithm (repository_pilot_20261112T003912Z.log). Results are therefore not claimed to be bitwise deterministic. The three final seeds describe variation and do not establish statistical significance.

Training uses CUDA BF16 autocast with FP32 model parameters and optimizer state. Validation and deployed prediction use the pipeline's default FP32 forward path outside autocast. Reported precision therefore describes training autocast, not a claim that every inference operation is BF16.

Classifier latency starts from prepared features. End-to-end inference latency includes tokenization/encoding plus prediction for processed input text; one-time dataset download/cleaning/splitting is measured separately. The configured latency procedure uses 10 warm-ups and 50 timed calls; throughput uses batch size 128. Device and CPU-thread fields must be consulted when comparing CPU TF-IDF and CUDA neural costs.


Recorded production child-process CPU hours: 9.673105. Completed GPU-assigned job wall hours: 5.970466; this is elapsed job time, not GPU utilization-hours.

Largest sampled allocated study storage: 19.041519 GiB, excluding archives and the installed environment. Smallest sampled filesystem free space: 193.594372 GiB. Sampling began 2026-11-12T01:22:53.965219+00:00 at five-minute intervals, so earlier and brief transient peaks may be missed. Archive byte size is recorded separately in archive_manifest.json.

Authoritative operational data: source/results/server/operational_totals.json, storage_peak.json, storage_history.csv and stage_timings.csv. The post-run report/audit were regenerated after the coordinator stopped; prior result archives are preserved.
