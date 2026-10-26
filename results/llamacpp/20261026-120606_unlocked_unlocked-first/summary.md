# llama.cpp run 20261026-120606_unlocked_unlocked-first

- Phase **unlocked**, gpu_unlock state **unlocked**
- llama.cpp fd8f8451a 2026-10-26 tests : refactor test-llama-archs to not link internal libllama symbols (#30880)
- Duration 7.0 min

| Test | Expect | Median tokens/s | CV % | n | Avg W | Max °C | Status |
|---|---|---:|---:|---:|---:|---:|---|
| BF16_pp512_fa-off | gain | 4612.4 | 0.5 | 7 | 37.6 | 56 | ok |
| BF16_pp2048_fa-off | gain | 3839.0 | 0.6 | 7 | 65.3 | 56 | ok |
| BF16_tg128_fa-off | control | 52.4 | 0.3 | 7 | 91.0 | 59 | ok |
| BF16_pp512_fa-on | gain | 5294.6 | 0.5 | 7 | 37.4 | 56 | ok |
| BF16_pp2048_fa-on | gain | 4983.1 | 1.9 | 7 | 63.5 | 55 | ok |
| BF16_tg128_fa-on | control | 53.3 | 0.2 | 7 | 93.0 | 58 | ok |
| F16_pp512_fa-off | info | 4719.3 | 0.7 | 7 | 33.5 | 55 | ok |
| F16_pp2048_fa-off | info | 3647.6 | 1.3 | 7 | 63.0 | 55 | ok |
| F16_tg128_fa-off | control | 52.5 | 0.2 | 7 | 92.2 | 59 | ok |
| F16_pp512_fa-on | info | 5432.1 | 3.0 | 7 | 42.4 | 56 | ok |
| F16_pp2048_fa-on | info | 5055.6 | 2.5 | 7 | 64.1 | 55 | ok |
| F16_tg128_fa-on | control | 53.5 | 0.3 | 7 | 90.8 | 58 | ok |
| Q8_0_pp512_fa-off | info | 5108.4 | 0.4 | 7 | 30.1 | 55 | ok |
| Q8_0_pp2048_fa-off | info | 4086.1 | 1.6 | 7 | 64.4 | 56 | ok |
| Q8_0_tg128_fa-off | control | 89.3 | 0.8 | 7 | 83.2 | 57 | ok |
| Q8_0_pp512_fa-on | info | 6014.1 | 0.3 | 7 | 35.6 | 54 | ok |
| Q8_0_pp2048_fa-on | info | 5515.3 | 3.1 | 7 | 66.5 | 54 | ok |
| Q8_0_tg128_fa-on | control | 92.8 | 0.6 | 7 | 84.9 | 56 | ok |
| Q4_K_M_pp512_fa-off | info | 4909.7 | 0.9 | 7 | 39.2 | 53 | ok |
| Q4_K_M_pp2048_fa-off | info | 3975.2 | 2.5 | 7 | 73.5 | 54 | ok |
| Q4_K_M_tg128_fa-off | control | 128.3 | 1.6 | 7 | 81.5 | 55 | ok |
| Q4_K_M_pp512_fa-on | info | 5401.2 | 0.2 | 7 | 37.1 | 53 | ok |
| Q4_K_M_pp2048_fa-on | info | 5235.1 | 0.6 | 7 | 70.7 | 53 | ok |
| Q4_K_M_tg128_fa-on | control | 134.6 | 1.0 | 7 | 81.3 | 54 | ok |
