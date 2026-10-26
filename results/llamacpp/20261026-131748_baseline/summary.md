# llama.cpp run 20261026-131748_baseline

- Phase **baseline**, gpu_unlock state **stock**
- llama.cpp fd8f8451a 2026-10-26 tests : refactor test-llama-archs to not link internal libllama symbols (#30880)
- Duration 7.0 min

| Test | Expect | Median tokens/s | CV % | n | Avg W | Max °C | Status |
|---|---|---:|---:|---:|---:|---:|---|
| BF16_pp512_fa-off | gain | 3699.2 | 1.6 | 7 | 35.1 | 56 | ok |
| BF16_pp2048_fa-off | gain | 3068.8 | 0.6 | 7 | 70.8 | 56 | ok |
| BF16_tg128_fa-off | control | 52.4 | 0.3 | 7 | 91.7 | 59 | ok |
| BF16_pp512_fa-on | gain | 4110.7 | 1.8 | 7 | 40.1 | 56 | ok |
| BF16_pp2048_fa-on | gain | 3901.5 | 1.7 | 7 | 69.9 | 56 | ok |
| BF16_tg128_fa-on | control | 53.3 | 0.3 | 7 | 91.8 | 59 | ok |
| F16_pp512_fa-off | info | 4666.0 | 0.5 | 7 | 30.8 | 55 | ok |
| F16_pp2048_fa-off | info | 3855.0 | 1.1 | 7 | 69.7 | 55 | ok |
| F16_tg128_fa-off | control | 52.6 | 0.3 | 7 | 92.4 | 58 | ok |
| F16_pp512_fa-on | info | 5422.6 | 0.6 | 7 | 34.5 | 55 | ok |
| F16_pp2048_fa-on | info | 5049.3 | 1.9 | 7 | 66.1 | 54 | ok |
| F16_tg128_fa-on | control | 53.5 | 0.3 | 7 | 91.5 | 57 | ok |
| Q8_0_pp512_fa-off | info | 4968.3 | 0.5 | 7 | 30.7 | 54 | ok |
| Q8_0_pp2048_fa-off | info | 4129.0 | 1.8 | 7 | 68.0 | 55 | ok |
| Q8_0_tg128_fa-off | control | 90.3 | 0.7 | 7 | 87.3 | 56 | ok |
| Q8_0_pp512_fa-on | info | 5943.8 | 0.6 | 7 | 34.9 | 53 | ok |
| Q8_0_pp2048_fa-on | info | 5448.5 | 2.9 | 7 | 67.5 | 53 | ok |
| Q8_0_tg128_fa-on | control | 92.5 | 0.7 | 7 | 84.3 | 55 | ok |
| Q4_K_M_pp512_fa-off | info | 4845.3 | 1.0 | 7 | 42.6 | 52 | ok |
| Q4_K_M_pp2048_fa-off | info | 3945.5 | 0.9 | 7 | 75.6 | 53 | ok |
| Q4_K_M_tg128_fa-off | control | 127.5 | 1.9 | 7 | 82.0 | 55 | ok |
| Q4_K_M_pp512_fa-on | info | 5570.7 | 0.3 | 7 | 39.3 | 54 | ok |
| Q4_K_M_pp2048_fa-on | info | 5118.8 | 2.9 | 7 | 73.2 | 53 | ok |
| Q4_K_M_tg128_fa-on | control | 134.2 | 1.5 | 7 | 82.1 | 55 | ok |
