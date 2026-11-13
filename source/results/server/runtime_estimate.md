# Measured pilot runtime estimate

{
  "generated_at": "2026-11-12T16:15:37.359457+00:00",
  "phase_hours": [
    {
      "phase": "bonus_final",
      "device": "cpu",
      "jobs": 0,
      "already_completed": 0,
      "low_hours": 0.0,
      "central_hours": 0.0,
      "high_hours": 0.0
    },
    {
      "phase": "bonus_final",
      "device": "cuda",
      "jobs": 108,
      "already_completed": 0,
      "low_hours": 0.5039034204842312,
      "central_hours": 1.5566257047020677,
      "high_hours": 3.722225832235901
    },
    {
      "phase": "bonus_tuning",
      "device": "cpu",
      "jobs": 0,
      "already_completed": 0,
      "low_hours": 0.0,
      "central_hours": 0.0,
      "high_hours": 0.0
    },
    {
      "phase": "bonus_tuning",
      "device": "cuda",
      "jobs": 36,
      "already_completed": 36,
      "low_hours": 0.0,
      "central_hours": 0.0,
      "high_hours": 0.0
    },
    {
      "phase": "part1_final",
      "device": "cpu",
      "jobs": 372,
      "already_completed": 372,
      "low_hours": 0.0,
      "central_hours": 0.0,
      "high_hours": 0.0
    },
    {
      "phase": "part1_final",
      "device": "cuda",
      "jobs": 465,
      "already_completed": 465,
      "low_hours": 0.0,
      "central_hours": 0.0,
      "high_hours": 0.0
    },
    {
      "phase": "part1_tuning",
      "device": "cpu",
      "jobs": 120,
      "already_completed": 120,
      "low_hours": 0.0,
      "central_hours": 0.0,
      "high_hours": 0.0
    },
    {
      "phase": "part1_tuning",
      "device": "cuda",
      "jobs": 150,
      "already_completed": 150,
      "low_hours": 0.0,
      "central_hours": 0.0,
      "high_hours": 0.0
    }
  ],
  "measured_stage_timings": [
    {
      "stage": "preflight",
      "started_at": "2026-11-12T00:37:33.816174+00:00",
      "finished_at": "2026-11-12T00:37:38.797921+00:00",
      "wall_sec": "4.981751506973524",
      "child_cpu_sec": "5.3590100000000005",
      "exit_code": "0",
      "log": "source/results/server/preflight_20261112T003733Z.log"
    },
    {
      "stage": "test",
      "started_at": "2026-11-12T00:37:38.816498+00:00",
      "finished_at": "2026-11-12T00:37:45.017665+00:00",
      "wall_sec": "6.201170996995643",
      "child_cpu_sec": "6.661078999999998",
      "exit_code": "0",
      "log": "source/results/server/test_20261112T003738Z.log"
    },
    {
      "stage": "server_tests",
      "started_at": "2026-11-12T00:37:45.035927+00:00",
      "finished_at": "2026-11-12T00:37:47.232659+00:00",
      "wall_sec": "2.196735747042112",
      "child_cpu_sec": "2.5714190000000015",
      "exit_code": "0",
      "log": "source/results/server/server_tests_20261112T003745Z.log"
    },
    {
      "stage": "download",
      "started_at": "2026-11-12T00:37:47.250833+00:00",
      "finished_at": "2026-11-12T00:37:51.229978+00:00",
      "wall_sec": "3.9791493359953165",
      "child_cpu_sec": "4.377879999999998",
      "exit_code": "0",
      "log": "source/results/server/download_20261112T003747Z.log"
    },
    {
      "stage": "preprocess",
      "started_at": "2026-11-12T00:37:51.247550+00:00",
      "finished_at": "2026-11-12T00:38:22.460869+00:00",
      "wall_sec": "31.21332253998844",
      "child_cpu_sec": "31.31416",
      "exit_code": "0",
      "log": "source/results/server/preprocess_20261112T003751Z.log"
    },
    {
      "stage": "plan",
      "started_at": "2026-11-12T00:38:22.473779+00:00",
      "finished_at": "2026-11-12T00:38:26.122511+00:00",
      "wall_sec": "3.6487360179889947",
      "child_cpu_sec": "4.050245999999996",
      "exit_code": "0",
      "log": "source/results/server/plan_20261112T003822Z.log"
    },
    {
      "stage": "smoke",
      "started_at": "2026-11-12T00:38:34.927596+00:00",
      "finished_at": "2026-11-12T00:38:41.373995+00:00",
      "wall_sec": "6.446403420006391",
      "child_cpu_sec": "7.387289",
      "exit_code": "0",
      "log": "source/results/server/smoke_20261112T003834Z.log"
    },
    {
      "stage": "memory_probe_avg_mlp",
      "started_at": "2026-11-12T00:38:42.953170+00:00",
      "finished_at": "2026-11-12T00:38:46.241988+00:00",
      "wall_sec": "3.28882271196926",
      "child_cpu_sec": "3.8611820000000003",
      "exit_code": "0",
      "log": "source/results/server/memory_probe_avg_mlp_20261112T003842Z.log"
    },
    {
      "stage": "memory_probe_lstm",
      "started_at": "2026-11-12T00:38:46.259152+00:00",
      "finished_at": "2026-11-12T00:38:49.694973+00:00",
      "wall_sec": "3.4358262150199153",
      "child_cpu_sec": "4.000596",
      "exit_code": "0",
      "log": "source/results/server/memory_probe_lstm_20261112T003846Z.log"
    },
    {
      "stage": "memory_probe_gru",
      "started_at": "2026-11-12T00:38:49.711929+00:00",
      "finished_at": "2026-11-12T00:38:53.115187+00:00",
      "wall_sec": "3.403263498970773",
      "child_cpu_sec": "3.9836059999999978",
      "exit_code": "0",
      "log": "source/results/server/memory_probe_gru_20261112T003849Z.log"
    },
    {
      "stage": "memory_probe_transformer",
      "started_at": "2026-11-12T00:38:53.132855+00:00",
      "finished_at": "2026-11-12T00:38:56.381531+00:00",
      "wall_sec": "3.248679689015262",
      "child_cpu_sec": "3.85331",
      "exit_code": "0",
      "log": "source/results/server/memory_probe_transformer_20261112T003853Z.log"
    },
    {
      "stage": "memory_probe_mamba",
      "started_at": "2026-11-12T00:38:56.398692+00:00",
      "finished_at": "2026-11-12T00:39:00.411814+00:00",
      "wall_sec": "4.013127104961313",
      "child_cpu_sec": "4.630552000000001",
      "exit_code": "0",
      "log": "source/results/server/memory_probe_mamba_20261112T003856Z.log"
    },
    {
      "stage": "memory_probe_pooled_ce",
      "started_at": "2026-11-12T00:39:00.428883+00:00",
      "finished_at": "2026-11-12T00:39:03.915038+00:00",
      "wall_sec": "3.4861604740144685",
      "child_cpu_sec": "4.0944160000000025",
      "exit_code": "0",
      "log": "source/results/server/memory_probe_pooled_ce_20261112T003900Z.log"
    },
    {
      "stage": "memory_probe_supcon",
      "started_at": "2026-11-12T00:39:03.932317+00:00",
      "finished_at": "2026-11-12T00:39:07.505907+00:00",
      "wall_sec": "3.5735947560169734",
      "child_cpu_sec": "4.158725999999996",
      "exit_code": "0",
      "log": "source/results/server/memory_probe_supcon_20261112T003903Z.log"
    },
    {
      "stage": "memory_probe_db_cdsupcon",
      "started_at": "2026-11-12T00:39:07.523929+00:00",
      "finished_at": "2026-11-12T00:39:11.058281+00:00",
      "wall_sec": "3.5343573709833436",
      "child_cpu_sec": "4.147840000000007",
      "exit_code": "0",
      "log": "source/results/server/memory_probe_db_cdsupcon_20261112T003907Z.log"
    },
    {
      "stage": "repository_pilot",
      "started_at": "2026-11-12T00:39:12.523610+00:00",
      "finished_at": "2026-11-12T00:39:31.482485+00:00",
      "wall_sec": "18.958879296958912",
      "child_cpu_sec": "20.306722999999998",
      "exit_code": "0",
      "log": "source/results/server/repository_pilot_20261112T003912Z.log"
    },
    {
      "stage": "pilot_nsmc_tfidf_lr_200",
      "started_at": "2026-11-12T00:39:31.503232+00:00",
      "finished_at": "2026-11-12T00:39:34.270352+00:00",
      "wall_sec": "2.7671236430178396",
      "child_cpu_sec": "3.5527959999999994",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_tfidf_lr_200_20261112T003931Z.log"
    },
    {
      "stage": "pilot_nsmc_tfidf_lr_8000",
      "started_at": "2026-11-12T00:39:34.287440+00:00",
      "finished_at": "2026-11-12T00:39:37.256041+00:00",
      "wall_sec": "2.9686045810231008",
      "child_cpu_sec": "4.016286000000001",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_tfidf_lr_8000_20261112T003934Z.log"
    },
    {
      "stage": "pilot_nsmc_tfidf_svm_200",
      "started_at": "2026-11-12T00:39:37.273899+00:00",
      "finished_at": "2026-11-12T00:39:39.946527+00:00",
      "wall_sec": "2.672631955007091",
      "child_cpu_sec": "3.459925",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_tfidf_svm_200_20261112T003937Z.log"
    },
    {
      "stage": "pilot_nsmc_tfidf_svm_8000",
      "started_at": "2026-11-12T00:39:39.963720+00:00",
      "finished_at": "2026-11-12T00:39:42.887823+00:00",
      "wall_sec": "2.924106256046798",
      "child_cpu_sec": "3.7150010000000058",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_tfidf_svm_8000_20261112T003939Z.log"
    },
    {
      "stage": "pilot_nsmc_tfidf_rf_200",
      "started_at": "2026-11-12T00:39:42.905066+00:00",
      "finished_at": "2026-11-12T00:39:48.551177+00:00",
      "wall_sec": "5.646114803967066",
      "child_cpu_sec": "5.918305999999998",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_tfidf_rf_200_20261112T003942Z.log"
    },
    {
      "stage": "pilot_nsmc_tfidf_rf_8000",
      "started_at": "2026-11-12T00:39:48.568606+00:00",
      "finished_at": "2026-11-12T00:39:54.481735+00:00",
      "wall_sec": "5.913133473019116",
      "child_cpu_sec": "6.785206000000004",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_tfidf_rf_8000_20261112T003948Z.log"
    },
    {
      "stage": "pilot_nsmc_tfidf_rf_130797",
      "started_at": "2026-11-12T00:39:54.498701+00:00",
      "finished_at": "2026-11-12T00:40:04.895453+00:00",
      "wall_sec": "10.396755940979347",
      "child_cpu_sec": "14.036915999999998",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_tfidf_rf_130797_20261112T003954Z.log"
    },
    {
      "stage": "pilot_nsmc_tfidf_xgb_200",
      "started_at": "2026-11-12T00:40:04.912718+00:00",
      "finished_at": "2026-11-12T00:40:07.745130+00:00",
      "wall_sec": "2.832416284014471",
      "child_cpu_sec": "4.082895999999996",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_tfidf_xgb_200_20261112T004004Z.log"
    },
    {
      "stage": "pilot_nsmc_tfidf_xgb_8000",
      "started_at": "2026-11-12T00:40:07.762926+00:00",
      "finished_at": "2026-11-12T00:40:11.233165+00:00",
      "wall_sec": "3.470239791960921",
      "child_cpu_sec": "5.8889099999999965",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_tfidf_xgb_8000_20261112T004007Z.log"
    },
    {
      "stage": "pilot_nsmc_avg_mlp_200",
      "started_at": "2026-11-12T00:40:11.250887+00:00",
      "finished_at": "2026-11-12T00:40:15.468294+00:00",
      "wall_sec": "4.2174113629735075",
      "child_cpu_sec": "5.06633800000001",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_avg_mlp_200_20261112T004011Z.log"
    },
    {
      "stage": "pilot_nsmc_avg_mlp_8000",
      "started_at": "2026-11-12T00:40:15.485279+00:00",
      "finished_at": "2026-11-12T00:40:20.654616+00:00",
      "wall_sec": "5.169340330990963",
      "child_cpu_sec": "6.201433000000001",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_avg_mlp_8000_20261112T004015Z.log"
    },
    {
      "stage": "pilot_nsmc_lstm_200",
      "started_at": "2026-11-12T00:40:20.672179+00:00",
      "finished_at": "2026-11-12T00:40:25.443212+00:00",
      "wall_sec": "4.7710374830057845",
      "child_cpu_sec": "5.662399999999993",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_lstm_200_20261112T004020Z.log"
    },
    {
      "stage": "pilot_nsmc_lstm_8000",
      "started_at": "2026-11-12T00:40:25.460360+00:00",
      "finished_at": "2026-11-12T00:40:32.432929+00:00",
      "wall_sec": "6.9725734720123",
      "child_cpu_sec": "8.018447000000005",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_lstm_8000_20261112T004025Z.log"
    },
    {
      "stage": "pilot_nsmc_gru_200",
      "started_at": "2026-11-12T00:40:32.450656+00:00",
      "finished_at": "2026-11-12T00:40:37.210757+00:00",
      "wall_sec": "4.760104612039868",
      "child_cpu_sec": "5.582811999999991",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_gru_200_20261112T004032Z.log"
    },
    {
      "stage": "pilot_nsmc_gru_8000",
      "started_at": "2026-11-12T00:40:37.228201+00:00",
      "finished_at": "2026-11-12T00:40:44.216275+00:00",
      "wall_sec": "6.988078216963913",
      "child_cpu_sec": "8.055942999999996",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_gru_8000_20261112T004037Z.log"
    },
    {
      "stage": "pilot_nsmc_transformer_200",
      "started_at": "2026-11-12T00:40:44.234572+00:00",
      "finished_at": "2026-11-12T00:40:48.937497+00:00",
      "wall_sec": "4.702929307008162",
      "child_cpu_sec": "5.594433000000002",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_transformer_200_20261112T004044Z.log"
    },
    {
      "stage": "pilot_nsmc_transformer_8000",
      "started_at": "2026-11-12T00:40:48.955519+00:00",
      "finished_at": "2026-11-12T00:40:55.151130+00:00",
      "wall_sec": "6.195614482974634",
      "child_cpu_sec": "7.140688000000001",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_transformer_8000_20261112T004048Z.log"
    },
    {
      "stage": "pilot_nsmc_transformer_130797",
      "started_at": "2026-11-12T00:40:55.168325+00:00",
      "finished_at": "2026-11-12T00:41:23.651479+00:00",
      "wall_sec": "28.483158405986615",
      "child_cpu_sec": "29.342632",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_transformer_130797_20261112T004055Z.log"
    },
    {
      "stage": "pilot_nsmc_mamba_200",
      "started_at": "2026-11-12T00:41:23.661810+00:00",
      "finished_at": "2026-11-12T00:41:28.825430+00:00",
      "wall_sec": "5.163624169013929",
      "child_cpu_sec": "6.128443999999989",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_mamba_200_20261112T004123Z.log"
    },
    {
      "stage": "pilot_nsmc_mamba_8000",
      "started_at": "2026-11-12T00:41:28.842673+00:00",
      "finished_at": "2026-11-12T00:41:35.429545+00:00",
      "wall_sec": "6.5868757260032",
      "child_cpu_sec": "7.68385000000001",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_mamba_8000_20261112T004128Z.log"
    },
    {
      "stage": "pilot_nsmc_mamba_130797",
      "started_at": "2026-11-12T00:41:35.447563+00:00",
      "finished_at": "2026-11-12T00:42:04.400762+00:00",
      "wall_sec": "28.953203915036283",
      "child_cpu_sec": "29.83622399999998",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_mamba_130797_20261112T004135Z.log"
    },
    {
      "stage": "pilot_nsmc_pooled_ce_500",
      "started_at": "2026-11-12T00:42:04.418751+00:00",
      "finished_at": "2026-11-12T00:42:11.092474+00:00",
      "wall_sec": "6.673724032996688",
      "child_cpu_sec": "7.621215000000003",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_pooled_ce_500_20261112T004204Z.log"
    },
    {
      "stage": "pilot_nsmc_supcon_500",
      "started_at": "2026-11-12T00:42:11.113645+00:00",
      "finished_at": "2026-11-12T00:42:18.529692+00:00",
      "wall_sec": "7.416051455016714",
      "child_cpu_sec": "8.37907499999999",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_supcon_500_20261112T004211Z.log"
    },
    {
      "stage": "pilot_nsmc_db_cdsupcon_500",
      "started_at": "2026-11-12T00:42:18.547111+00:00",
      "finished_at": "2026-11-12T00:42:25.784467+00:00",
      "wall_sec": "7.237359655031469",
      "child_cpu_sec": "8.16766100000002",
      "exit_code": "0",
      "log": "source/results/server/pilot_nsmc_db_cdsupcon_500_20261112T004218Z.log"
    },
    {
      "stage": "pilot_shopping_tfidf_lr_200",
      "started_at": "2026-11-12T00:42:25.801624+00:00",
      "finished_at": "2026-11-12T00:42:28.693089+00:00",
      "wall_sec": "2.891481668979395",
      "child_cpu_sec": "3.815266000000012",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_tfidf_lr_200_20261112T004225Z.log"
    },
    {
      "stage": "pilot_shopping_tfidf_lr_8000",
      "started_at": "2026-11-12T00:42:28.710348+00:00",
      "finished_at": "2026-11-12T00:42:31.897649+00:00",
      "wall_sec": "3.187304717954248",
      "child_cpu_sec": "4.20103200000001",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_tfidf_lr_8000_20261112T004228Z.log"
    },
    {
      "stage": "pilot_shopping_tfidf_svm_200",
      "started_at": "2026-11-12T00:42:31.915100+00:00",
      "finished_at": "2026-11-12T00:42:34.877685+00:00",
      "wall_sec": "2.9625888949958608",
      "child_cpu_sec": "3.803957000000006",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_tfidf_svm_200_20261112T004231Z.log"
    },
    {
      "stage": "pilot_shopping_tfidf_svm_8000",
      "started_at": "2026-11-12T00:42:34.895241+00:00",
      "finished_at": "2026-11-12T00:42:38.044103+00:00",
      "wall_sec": "3.1488663099589758",
      "child_cpu_sec": "3.9600150000000003",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_tfidf_svm_8000_20261112T004234Z.log"
    },
    {
      "stage": "pilot_shopping_tfidf_rf_200",
      "started_at": "2026-11-12T00:42:38.054711+00:00",
      "finished_at": "2026-11-12T00:42:43.925617+00:00",
      "wall_sec": "5.870910947967786",
      "child_cpu_sec": "6.164648999999981",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_tfidf_rf_200_20261112T004238Z.log"
    },
    {
      "stage": "pilot_shopping_tfidf_rf_8000",
      "started_at": "2026-11-12T00:42:43.943061+00:00",
      "finished_at": "2026-11-12T00:42:50.082731+00:00",
      "wall_sec": "6.139673180994578",
      "child_cpu_sec": "7.006541000000015",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_tfidf_rf_8000_20261112T004243Z.log"
    },
    {
      "stage": "pilot_shopping_tfidf_rf_159918",
      "started_at": "2026-11-12T00:42:50.100520+00:00",
      "finished_at": "2026-11-12T00:43:02.252479+00:00",
      "wall_sec": "12.151963213982526",
      "child_cpu_sec": "17.051122",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_tfidf_rf_159918_20261112T004250Z.log"
    },
    {
      "stage": "pilot_shopping_tfidf_xgb_200",
      "started_at": "2026-11-12T00:43:02.270808+00:00",
      "finished_at": "2026-11-12T00:43:05.447038+00:00",
      "wall_sec": "3.176234833023045",
      "child_cpu_sec": "4.767472999999999",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_tfidf_xgb_200_20261112T004302Z.log"
    },
    {
      "stage": "pilot_shopping_tfidf_xgb_8000",
      "started_at": "2026-11-12T00:43:05.464288+00:00",
      "finished_at": "2026-11-12T00:43:09.198046+00:00",
      "wall_sec": "3.7337615590076894",
      "child_cpu_sec": "6.287974000000002",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_tfidf_xgb_8000_20261112T004305Z.log"
    },
    {
      "stage": "pilot_shopping_avg_mlp_200",
      "started_at": "2026-11-12T00:43:09.215271+00:00",
      "finished_at": "2026-11-12T00:43:13.653971+00:00",
      "wall_sec": "4.438704741012771",
      "child_cpu_sec": "5.295273999999933",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_avg_mlp_200_20261112T004309Z.log"
    },
    {
      "stage": "pilot_shopping_avg_mlp_8000",
      "started_at": "2026-11-12T00:43:13.671065+00:00",
      "finished_at": "2026-11-12T00:43:19.154686+00:00",
      "wall_sec": "5.483626052970067",
      "child_cpu_sec": "6.425651000000002",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_avg_mlp_8000_20261112T004313Z.log"
    },
    {
      "stage": "pilot_shopping_lstm_200",
      "started_at": "2026-11-12T00:43:19.172210+00:00",
      "finished_at": "2026-11-12T00:43:24.272295+00:00",
      "wall_sec": "5.100089156010654",
      "child_cpu_sec": "5.9789669999999955",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_lstm_200_20261112T004319Z.log"
    },
    {
      "stage": "pilot_shopping_lstm_8000",
      "started_at": "2026-11-12T00:43:24.293117+00:00",
      "finished_at": "2026-11-12T00:43:31.679831+00:00",
      "wall_sec": "7.386718063033186",
      "child_cpu_sec": "8.483257000000041",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_lstm_8000_20261112T004324Z.log"
    },
    {
      "stage": "pilot_shopping_gru_200",
      "started_at": "2026-11-12T00:43:31.697017+00:00",
      "finished_at": "2026-11-12T00:43:36.850739+00:00",
      "wall_sec": "5.15372582199052",
      "child_cpu_sec": "6.108264999999978",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_gru_200_20261112T004331Z.log"
    },
    {
      "stage": "pilot_shopping_gru_8000",
      "started_at": "2026-11-12T00:43:36.867995+00:00",
      "finished_at": "2026-11-12T00:43:44.267829+00:00",
      "wall_sec": "7.399836941040121",
      "child_cpu_sec": "8.542631999999944",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_gru_8000_20261112T004336Z.log"
    },
    {
      "stage": "pilot_shopping_transformer_200",
      "started_at": "2026-11-12T00:43:44.285613+00:00",
      "finished_at": "2026-11-12T00:43:49.395253+00:00",
      "wall_sec": "5.109643984993454",
      "child_cpu_sec": "5.961795000000027",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_transformer_200_20261112T004344Z.log"
    },
    {
      "stage": "pilot_shopping_transformer_8000",
      "started_at": "2026-11-12T00:43:49.412656+00:00",
      "finished_at": "2026-11-12T00:43:56.172101+00:00",
      "wall_sec": "6.759448751981836",
      "child_cpu_sec": "7.75362999999998",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_transformer_8000_20261112T004349Z.log"
    },
    {
      "stage": "pilot_shopping_transformer_159918",
      "started_at": "2026-11-12T00:43:56.189185+00:00",
      "finished_at": "2026-11-12T00:44:30.774440+00:00",
      "wall_sec": "34.58525902504334",
      "child_cpu_sec": "35.26485600000002",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_transformer_159918_20261112T004356Z.log"
    },
    {
      "stage": "pilot_shopping_mamba_200",
      "started_at": "2026-11-12T00:44:30.792661+00:00",
      "finished_at": "2026-11-12T00:44:36.436142+00:00",
      "wall_sec": "5.643486474989913",
      "child_cpu_sec": "6.579266999999998",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_mamba_200_20261112T004430Z.log"
    },
    {
      "stage": "pilot_shopping_mamba_8000",
      "started_at": "2026-11-12T00:44:36.453834+00:00",
      "finished_at": "2026-11-12T00:44:43.470876+00:00",
      "wall_sec": "7.017045360989869",
      "child_cpu_sec": "8.223701999999943",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_mamba_8000_20261112T004436Z.log"
    },
    {
      "stage": "pilot_shopping_mamba_159918",
      "started_at": "2026-11-12T00:44:43.488164+00:00",
      "finished_at": "2026-11-12T00:45:18.286475+00:00",
      "wall_sec": "34.798315728025045",
      "child_cpu_sec": "35.70070600000001",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_mamba_159918_20261112T004443Z.log"
    },
    {
      "stage": "pilot_shopping_pooled_ce_500",
      "started_at": "2026-11-12T00:45:18.303602+00:00",
      "finished_at": "2026-11-12T00:45:25.422328+00:00",
      "wall_sec": "7.118731498951092",
      "child_cpu_sec": "8.030633000000012",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_pooled_ce_500_20261112T004518Z.log"
    },
    {
      "stage": "pilot_shopping_supcon_500",
      "started_at": "2026-11-12T00:45:25.439937+00:00",
      "finished_at": "2026-11-12T00:45:32.991357+00:00",
      "wall_sec": "7.55142336001154",
      "child_cpu_sec": "8.555906000000004",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_supcon_500_20261112T004525Z.log"
    },
    {
      "stage": "pilot_shopping_db_cdsupcon_500",
      "started_at": "2026-11-12T00:45:33.020521+00:00",
      "finished_at": "2026-11-12T00:45:40.663485+00:00",
      "wall_sec": "7.642967778025195",
      "child_cpu_sec": "8.680458000000073",
      "exit_code": "0",
      "log": "source/results/server/pilot_shopping_db_cdsupcon_500_20261112T004533Z.log"
    },
    {
      "stage": "pilot_steam_tfidf_lr_200",
      "started_at": "2026-11-12T00:45:40.681175+00:00",
      "finished_at": "2026-11-12T00:45:43.144856+00:00",
      "wall_sec": "2.4636860080063343",
      "child_cpu_sec": "3.341034999999998",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_tfidf_lr_200_20261112T004540Z.log"
    },
    {
      "stage": "pilot_steam_tfidf_lr_8000",
      "started_at": "2026-11-12T00:45:43.163007+00:00",
      "finished_at": "2026-11-12T00:45:45.932230+00:00",
      "wall_sec": "2.769227715965826",
      "child_cpu_sec": "3.8005809999999727",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_tfidf_lr_8000_20261112T004543Z.log"
    },
    {
      "stage": "pilot_steam_tfidf_svm_200",
      "started_at": "2026-11-12T00:45:45.943155+00:00",
      "finished_at": "2026-11-12T00:45:48.387268+00:00",
      "wall_sec": "2.444117447012104",
      "child_cpu_sec": "3.230011999999988",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_tfidf_svm_200_20261112T004545Z.log"
    },
    {
      "stage": "pilot_steam_tfidf_svm_8000",
      "started_at": "2026-11-12T00:45:48.405029+00:00",
      "finished_at": "2026-11-12T00:45:51.149236+00:00",
      "wall_sec": "2.7442116049933247",
      "child_cpu_sec": "3.6171889999999784",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_tfidf_svm_8000_20261112T004548Z.log"
    },
    {
      "stage": "pilot_steam_tfidf_rf_200",
      "started_at": "2026-11-12T00:45:51.166668+00:00",
      "finished_at": "2026-11-12T00:45:56.534308+00:00",
      "wall_sec": "5.36764365201816",
      "child_cpu_sec": "5.644023999999984",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_tfidf_rf_200_20261112T004551Z.log"
    },
    {
      "stage": "pilot_steam_tfidf_rf_8000",
      "started_at": "2026-11-12T00:45:56.551860+00:00",
      "finished_at": "2026-11-12T00:46:02.297702+00:00",
      "wall_sec": "5.745846297999378",
      "child_cpu_sec": "6.559648000000042",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_tfidf_rf_8000_20261112T004556Z.log"
    },
    {
      "stage": "pilot_steam_tfidf_rf_79828",
      "started_at": "2026-11-12T00:46:02.315491+00:00",
      "finished_at": "2026-11-12T00:46:11.387107+00:00",
      "wall_sec": "9.07162043597782",
      "child_cpu_sec": "11.725099999999951",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_tfidf_rf_79828_20261112T004602Z.log"
    },
    {
      "stage": "pilot_steam_tfidf_xgb_200",
      "started_at": "2026-11-12T00:46:11.404514+00:00",
      "finished_at": "2026-11-12T00:46:14.094522+00:00",
      "wall_sec": "2.6900137849734165",
      "child_cpu_sec": "4.02252600000001",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_tfidf_xgb_200_20261112T004611Z.log"
    },
    {
      "stage": "pilot_steam_tfidf_xgb_8000",
      "started_at": "2026-11-12T00:46:14.111846+00:00",
      "finished_at": "2026-11-12T00:46:17.538873+00:00",
      "wall_sec": "3.4270311419968493",
      "child_cpu_sec": "6.197190999999961",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_tfidf_xgb_8000_20261112T004614Z.log"
    },
    {
      "stage": "pilot_steam_avg_mlp_200",
      "started_at": "2026-11-12T00:46:17.556340+00:00",
      "finished_at": "2026-11-12T00:46:21.472156+00:00",
      "wall_sec": "3.915819576010108",
      "child_cpu_sec": "4.818044000000029",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_avg_mlp_200_20261112T004617Z.log"
    },
    {
      "stage": "pilot_steam_avg_mlp_8000",
      "started_at": "2026-11-12T00:46:21.490178+00:00",
      "finished_at": "2026-11-12T00:46:26.365652+00:00",
      "wall_sec": "4.87547965196427",
      "child_cpu_sec": "5.828613999999998",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_avg_mlp_8000_20261112T004621Z.log"
    },
    {
      "stage": "pilot_steam_lstm_200",
      "started_at": "2026-11-12T00:46:26.383107+00:00",
      "finished_at": "2026-11-12T00:46:30.682645+00:00",
      "wall_sec": "4.299541441956535",
      "child_cpu_sec": "5.22010499999994",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_lstm_200_20261112T004626Z.log"
    },
    {
      "stage": "pilot_steam_lstm_8000",
      "started_at": "2026-11-12T00:46:30.701082+00:00",
      "finished_at": "2026-11-12T00:46:37.423652+00:00",
      "wall_sec": "6.722574380983133",
      "child_cpu_sec": "7.880080000000056",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_lstm_8000_20261112T004630Z.log"
    },
    {
      "stage": "pilot_steam_gru_200",
      "started_at": "2026-11-12T00:46:37.441148+00:00",
      "finished_at": "2026-11-12T00:46:41.800296+00:00",
      "wall_sec": "4.3591514700092375",
      "child_cpu_sec": "5.331695999999987",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_gru_200_20261112T004637Z.log"
    },
    {
      "stage": "pilot_steam_gru_8000",
      "started_at": "2026-11-12T00:46:41.819646+00:00",
      "finished_at": "2026-11-12T00:46:48.482812+00:00",
      "wall_sec": "6.66317017597612",
      "child_cpu_sec": "7.5761260000000235",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_gru_8000_20261112T004641Z.log"
    },
    {
      "stage": "pilot_steam_transformer_200",
      "started_at": "2026-11-12T00:46:48.502594+00:00",
      "finished_at": "2026-11-12T00:46:52.740302+00:00",
      "wall_sec": "4.237712828035001",
      "child_cpu_sec": "5.175316000000006",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_transformer_200_20261112T004648Z.log"
    },
    {
      "stage": "pilot_steam_transformer_8000",
      "started_at": "2026-11-12T00:46:52.760026+00:00",
      "finished_at": "2026-11-12T00:46:58.501022+00:00",
      "wall_sec": "5.74100115703186",
      "child_cpu_sec": "6.63267799999991",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_transformer_8000_20261112T004652Z.log"
    },
    {
      "stage": "pilot_steam_transformer_79828",
      "started_at": "2026-11-12T00:46:58.519481+00:00",
      "finished_at": "2026-11-12T00:47:17.546546+00:00",
      "wall_sec": "19.02706897200551",
      "child_cpu_sec": "19.964473000000066",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_transformer_79828_20261112T004658Z.log"
    },
    {
      "stage": "pilot_steam_mamba_200",
      "started_at": "2026-11-12T00:47:17.564661+00:00",
      "finished_at": "2026-11-12T00:47:22.329398+00:00",
      "wall_sec": "4.7647406999603845",
      "child_cpu_sec": "5.627351000000015",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_mamba_200_20261112T004717Z.log"
    },
    {
      "stage": "pilot_steam_mamba_8000",
      "started_at": "2026-11-12T00:47:22.347807+00:00",
      "finished_at": "2026-11-12T00:47:28.646223+00:00",
      "wall_sec": "6.298419967002701",
      "child_cpu_sec": "7.563960999999971",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_mamba_8000_20261112T004722Z.log"
    },
    {
      "stage": "pilot_steam_mamba_79828",
      "started_at": "2026-11-12T00:47:28.664875+00:00",
      "finished_at": "2026-11-12T00:47:48.087948+00:00",
      "wall_sec": "19.4230779579957",
      "child_cpu_sec": "20.361055000000068",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_mamba_79828_20261112T004728Z.log"
    },
    {
      "stage": "pilot_steam_pooled_ce_500",
      "started_at": "2026-11-12T00:47:48.107970+00:00",
      "finished_at": "2026-11-12T00:47:54.525878+00:00",
      "wall_sec": "6.417908609029837",
      "child_cpu_sec": "7.4943279999999035",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_pooled_ce_500_20261112T004748Z.log"
    },
    {
      "stage": "pilot_steam_supcon_500",
      "started_at": "2026-11-12T00:47:54.545449+00:00",
      "finished_at": "2026-11-12T00:48:01.398552+00:00",
      "wall_sec": "6.853107443021145",
      "child_cpu_sec": "7.8552770000001",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_supcon_500_20261112T004754Z.log"
    },
    {
      "stage": "pilot_steam_db_cdsupcon_500",
      "started_at": "2026-11-12T00:48:01.417230+00:00",
      "finished_at": "2026-11-12T00:48:08.294147+00:00",
      "wall_sec": "6.876921760034747",
      "child_cpu_sec": "7.829888999999973",
      "exit_code": "0",
      "log": "source/results/server/pilot_steam_db_cdsupcon_500_20261112T004801Z.log"
    },
    {
      "stage": "forecast_tests",
      "started_at": "2026-11-12T00:48:11.784476+00:00",
      "finished_at": "2026-11-12T00:48:12.685717+00:00",
      "wall_sec": "0.9012439939542674",
      "child_cpu_sec": "1.3367120000000001",
      "exit_code": "0",
      "log": "source/results/server/forecast_tests_20261112T004811Z.log"
    },
    {
      "stage": "tune",
      "started_at": "2026-11-12T00:48:12.913537+00:00",
      "finished_at": "2026-11-12T01:33:30.452526+00:00",
      "wall_sec": "2717.538994434988",
      "child_cpu_sec": "3475.6009750000003",
      "exit_code": "0",
      "log": "source/results/server/tune_20261112T004812Z.log"
    },
    {
      "stage": "full",
      "started_at": "2026-11-12T01:33:30.786490+00:00",
      "finished_at": "2026-11-12T04:48:47.590882+00:00",
      "wall_sec": "11716.804396904015",
      "child_cpu_sec": "17558.809931999996",
      "exit_code": "139",
      "log": "source/results/server/full_20261112T013330Z.log"
    },
    {
      "stage": "archive-results",
      "started_at": "2026-11-12T04:48:48.609325+00:00",
      "finished_at": "2026-11-12T04:49:11.763134+00:00",
      "wall_sec": "23.153812769975048",
      "child_cpu_sec": "23.23320599999711",
      "exit_code": "0",
      "log": "source/results/server/archive-results_20261112T044848Z.log"
    },
    {
      "stage": "forecast_tests",
      "started_at": "2026-11-12T13:48:58.459404+00:00",
      "finished_at": "2026-11-12T13:48:59.461751+00:00",
      "wall_sec": "1.0023508580052294",
      "child_cpu_sec": "1.422273",
      "exit_code": "0",
      "log": "source/results/server/forecast_tests_20261112T134858Z.log"
    },
    {
      "stage": "tune",
      "started_at": "2026-11-12T13:48:59.839199+00:00",
      "finished_at": "2026-11-12T13:49:03.636601+00:00",
      "wall_sec": "3.797407639969606",
      "child_cpu_sec": "4.180014",
      "exit_code": "0",
      "log": "source/results/server/tune_20261112T134859Z.log"
    },
    {
      "stage": "full",
      "started_at": "2026-11-12T13:49:04.167651+00:00",
      "finished_at": "2026-11-12T16:02:44.127982+00:00",
      "wall_sec": "8019.960335731972",
      "child_cpu_sec": "9932.128815",
      "exit_code": "0",
      "log": "source/results/server/full_20261112T134904Z.log"
    },
    {
      "stage": "aggregate",
      "started_at": "2026-11-12T16:02:44.380124+00:00",
      "finished_at": "2026-11-12T16:02:48.194857+00:00",
      "wall_sec": "3.8147378389840014",
      "child_cpu_sec": "4.165643999997997",
      "exit_code": "0",
      "log": "source/results/server/aggregate_20261112T160244Z.log"
    },
    {
      "stage": "bonus-plan",
      "started_at": "2026-11-12T16:02:48.447075+00:00",
      "finished_at": "2026-11-12T16:02:52.026238+00:00",
      "wall_sec": "3.5791674950160086",
      "child_cpu_sec": "3.9784030000002844",
      "exit_code": "0",
      "log": "source/results/server/bonus-plan_20261112T160248Z.log"
    },
    {
      "stage": "bonus-tune",
      "started_at": "2026-11-12T16:02:52.638804+00:00",
      "finished_at": "2026-11-12T16:15:36.747191+00:00",
      "wall_sec": "764.1083900899976",
      "child_cpu_sec": "779.0955770000004",
      "exit_code": "0",
      "log": "source/results/server/bonus-tune_20261112T160252Z.log"
    }
  ],
  "repository_pilot_records": [
    {
      "artifact_directory": "/home/dung/PracticalMachineLearningProject/source/results/run_records/part1-fdf4844c482946017ebc",
      "best_epoch": 11,
      "config_hash": "98fb46ab20274bd9e7b58e1aa86c734624a614a1dd1dd5b598fae7e394627d5f",
      "cpu_threads": 4,
      "data_manifest_hash": "7dc24428211cfe790a5c99d88b05317714d909883bb775e33826535f144344e7",
      "device": "cuda",
      "duration_sec": 15.145700112974737,
      "exact_command": "/home/dung/PracticalMachineLearningProject/source/src/korean_sentiment/cli.py pilot --dataset nsmc --model transformer --max-jobs 1 --device cuda",
      "finished_at": "2026-11-12T00:39:30.881600+00:00",
      "fit_time_sec": 14.513021521968767,
      "hardware_id": "8b5663550ab8bee2",
      "precision": "bf16",
      "run_id": "part1-fdf4844c482946017ebc",
      "serialized_size_mb": 8.081411361694336,
      "started_at": "2026-11-12T00:39:15.735914+00:00"
    },
    {
      "artifact_directory": "/home/dung/PracticalMachineLearningProject/source/results/run_records/part1-5b8a8ba58e414a1423ac",
      "best_epoch": 19,
      "config_hash": "896e042984367922dcff63930c320faa79cdca5d15cf24bf0a8abe20a79ce0dc",
      "cpu_threads": 4,
      "data_manifest_hash": "7dc24428211cfe790a5c99d88b05317714d909883bb775e33826535f144344e7",
      "device": "cuda",
      "duration_sec": 18.1422284609871,
      "exact_command": "/home/dung/PracticalMachineLearningProject/source/src/korean_sentiment/cli.py tune --fail-fast",
      "finished_at": "2026-11-12T00:58:02.765977+00:00",
      "fit_time_sec": 17.67423893196974,
      "hardware_id": "8b5663550ab8bee2",
      "precision": "bf16",
      "run_id": "part1-5b8a8ba58e414a1423ac",
      "serialized_size_mb": 8.081594467163086,
      "started_at": "2026-11-12T00:57:44.623762+00:00"
    },
    {
      "artifact_directory": "/home/dung/PracticalMachineLearningProject/source/results/run_records/part1-d12c5876d3abc0c5b077",
      "best_epoch": 19,
      "config_hash": "0340460d44ba615a80579eb062a8c4276f7975c8e2f7a966a2b0ae49dc3e234b",
      "cpu_threads": 4,
      "data_manifest_hash": "7dc24428211cfe790a5c99d88b05317714d909883bb775e33826535f144344e7",
      "device": "cuda",
      "duration_sec": 18.547296134987846,
      "exact_command": "/home/dung/PracticalMachineLearningProject/source/src/korean_sentiment/cli.py tune --fail-fast",
      "finished_at": "2026-11-12T00:58:21.356460+00:00",
      "fit_time_sec": 18.067025012976956,
      "hardware_id": "8b5663550ab8bee2",
      "precision": "bf16",
      "run_id": "part1-d12c5876d3abc0c5b077",
      "serialized_size_mb": 8.152273178100586,
      "started_at": "2026-11-12T00:58:02.809180+00:00"
    },
    {
      "artifact_directory": "/home/dung/PracticalMachineLearningProject/source/results/run_records/part1-df23f30411d30e1a5a42",
      "best_epoch": 15,
      "config_hash": "cd62dd8b38c142ffccbd5bd77bf0245befd3c7f00e574d0da9f48746ed873989",
      "cpu_threads": 4,
      "data_manifest_hash": "7dc24428211cfe790a5c99d88b05317714d909883bb775e33826535f144344e7",
      "device": "cuda",
      "duration_sec": 17.342312280030455,
      "exact_command": "/home/dung/PracticalMachineLearningProject/source/src/korean_sentiment/cli.py tune --fail-fast",
      "finished_at": "2026-11-12T00:58:38.740337+00:00",
      "fit_time_sec": 16.869486634037457,
      "hardware_id": "8b5663550ab8bee2",
      "precision": "bf16",
      "run_id": "part1-df23f30411d30e1a5a42",
      "serialized_size_mb": 8.081594467163086,
      "started_at": "2026-11-12T00:58:21.398040+00:00"
    },
    {
      "artifact_directory": "/home/dung/PracticalMachineLearningProject/source/results/run_records/part1-2989dd5c817e087d6cc3",
      "best_epoch": 17,
      "config_hash": "8e9488ef94c8d79ed046f0f36ab02972a7f313f666593b6e543b879f05774be6",
      "cpu_threads": 4,
      "data_manifest_hash": "7dc24428211cfe790a5c99d88b05317714d909883bb775e33826535f144344e7",
      "device": "cuda",
      "duration_sec": 29.179446767026093,
      "exact_command": "/home/dung/PracticalMachineLearningProject/source/src/korean_sentiment/cli.py tune --fail-fast",
      "finished_at": "2026-11-12T00:59:07.963180+00:00",
      "fit_time_sec": 28.44183280901052,
      "hardware_id": "8b5663550ab8bee2",
      "precision": "bf16",
      "run_id": "part1-2989dd5c817e087d6cc3",
      "serialized_size_mb": 11.268278121948242,
      "started_at": "2026-11-12T00:58:38.783748+00:00"
    },
    {
      "artifact_directory": "/home/dung/PracticalMachineLearningProject/source/results/run_records/part1-2ee765641ee7dfe1b460",
      "best_epoch": 19,
      "config_hash": "408f9f7f0b9d7cc1f10be61acccc760bb00911f4ceffe94b30771c1ed2fb73b5",
      "cpu_threads": 4,
      "data_manifest_hash": "7dc24428211cfe790a5c99d88b05317714d909883bb775e33826535f144344e7",
      "device": "cuda",
      "duration_sec": 29.069817556010094,
      "exact_command": "/home/dung/PracticalMachineLearningProject/source/src/korean_sentiment/cli.py tune --fail-fast",
      "finished_at": "2026-11-12T00:59:37.070833+00:00",
      "fit_time_sec": 28.32915409898851,
      "hardware_id": "8b5663550ab8bee2",
      "precision": "bf16",
      "run_id": "part1-2ee765641ee7dfe1b460",
      "serialized_size_mb": 11.519254684448242,
      "started_at": "2026-11-12T00:59:08.001030+00:00"
    },
    {
      "artifact_directory": "/home/dung/PracticalMachineLearningProject/source/results/run_records/part1-5ea8b7e804e5e3c487e9",
      "best_epoch": 20,
      "config_hash": "c39c16d675ea38e2c20de8f135f789a24765fb5e0f3995152532423e4e85ec2a",
      "cpu_threads": 4,
      "data_manifest_hash": "7dc24428211cfe790a5c99d88b05317714d909883bb775e33826535f144344e7",
      "device": "cuda",
      "duration_sec": 29.009442534006666,
      "exact_command": "/home/dung/PracticalMachineLearningProject/source/src/korean_sentiment/cli.py tune --fail-fast",
      "finished_at": "2026-11-12T01:00:06.118121+00:00",
      "fit_time_sec": 28.276397901994642,
      "hardware_id": "8b5663550ab8bee2",
      "precision": "bf16",
      "run_id": "part1-5ea8b7e804e5e3c487e9",
      "serialized_size_mb": 11.268278121948242,
      "started_at": "2026-11-12T00:59:37.108693+00:00"
    },
    {
      "artifact_directory": "/home/dung/PracticalMachineLearningProject/source/results/run_records/part1-ec25e1eb0e2c028ec133",
      "best_epoch": 8,
      "config_hash": "a8fe3f037434e2fd074014ccd2c55a186bcbd4cecbbebc051ff751669d60d070",
      "cpu_threads": 4,
      "data_manifest_hash": "7dc24428211cfe790a5c99d88b05317714d909883bb775e33826535f144344e7",
      "device": "cuda",
      "duration_sec": 19.000336736033205,
      "exact_command": "/home/dung/PracticalMachineLearningProject/source/src/korean_sentiment/cli.py tune --fail-fast",
      "finished_at": "2026-11-12T01:00:25.159166+00:00",
      "fit_time_sec": 18.175831319997087,
      "hardware_id": "8b5663550ab8bee2",
      "precision": "bf16",
      "run_id": "part1-ec25e1eb0e2c028ec133",
      "serialized_size_mb": 11.267972946166992,
      "started_at": "2026-11-12T01:00:06.158845+00:00"
    },
    {
      "artifact_directory": "/home/dung/PracticalMachineLearningProject/source/results/run_records/part1-08638d2f41d4d8a555bd",
      "best_epoch": 18,
      "config_hash": "b1c17db1cb8e0fb659f2618d66ee2f9edd568b7e03034bf267a7e6011a11c123",
      "cpu_threads": 4,
      "data_manifest_hash": "7dc24428211cfe790a5c99d88b05317714d909883bb775e33826535f144344e7",
      "device": "cuda",
      "duration_sec": 39.03548377397237,
      "exact_command": "/home/dung/PracticalMachineLearningProject/source/src/korean_sentiment/cli.py tune --fail-fast",
      "finished_at": "2026-11-12T01:01:04.227800+00:00",
      "fit_time_sec": 38.025784972007386,
      "hardware_id": "8b5663550ab8bee2",
      "precision": "bf16",
      "run_id": "part1-08638d2f41d4d8a555bd",
      "serialized_size_mb": 14.290983200073242,
      "started_at": "2026-11-12T01:00:25.192331+00:00"
    },
    {
      "artifact_directory": "/home/dung/PracticalMachineLearningProject/source/results/run_records/part1-3bc7d5ffedadf9d205ff",
      "best_epoch": 15,
      "config_hash": "adde47e1ae8bdd5f30175c3bac1649e2666ccccc565727ec05418af28db03814",
      "cpu_threads": 4,
      "data_manifest_hash": "7dc24428211cfe790a5c99d88b05317714d909883bb775e33826535f144344e7",
      "device": "cuda",
      "duration_sec": 51.06380295497365,
      "exact_command": "/home/dung/PracticalMachineLearningProject/source/src/korean_sentiment/cli.py tune --fail-fast",
      "finished_at": "2026-11-12T01:01:55.331279+00:00",
      "fit_time_sec": 49.73317168204812,
      "hardware_id": "8b5663550ab8bee2",
      "precision": "bf16",
      "run_id": "part1-3bc7d5ffedadf9d205ff",
      "serialized_size_mb": 15.082586288452148,
      "started_at": "2026-11-12T01:01:04.267491+00:00"
    }
  ],
  "wall_seconds": {
    "low": 1904.7549294303938,
    "central": 5884.045163773816,
    "high": 14070.013645851706
  },
  "completion_asia_seoul": {
    "low": "2026-11-13T01:47:22.115127+09:00",
    "central": "2026-11-13T02:53:41.405373+09:00",
    "high": "2026-11-13T05:10:07.373858+09:00"
  },
  "central_exceeds_24h": false,
  "safe_concurrency": 1,
  "assumptions": "Per-dataset/model/objective strata; monotone piecewise-linear size curves, separating measured neural per-epoch fit from loading/serialization/validation overhead. Neural 5/12/20-epoch scenarios; configuration-derived operation proxies for known architectures, or min/mean/max candidate proxies for unfrozen selections, with 0.65/1/1.5 GPU-efficiency uncertainty. Operation proxies are assumptions, not measured GPU FLOPs. Classical tree scaling and 0.75/1.25/2.5 uncertainty; final test overhead estimated from measured validation; 5% aggregation/monitor reserve. CPU/CUDA hours are assigned-device elapsed hours, not measured utilization-hours. Full-size pilots use RF, Transformer and Mamba candidate1. Other full-size families are extrapolated; hyperparameter depth/width and early stopping remain major uncertainty.",
  "comparison_5_9_days": "The initial 5\u20139 days is not an input to this calculation. Differences follow measured pilot strata, dataset sizes, early-stopping assumptions and conservative width/depth uncertainty."
}

Pilot details: runtime_estimate.json and server/pilots/. Per-run predictions: runtime_estimate.csv. Test overhead is an estimate based on validation; actual test evaluation remains restricted to final runs.
