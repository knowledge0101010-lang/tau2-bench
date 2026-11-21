# Editions

Omit `--profile` and `tau2 run` keeps the current defaults (`xai/grok-4.7` for text roles, `xai/grok-4.20-0309-non-reasoning` for the tool mentor). xAI Grok (`api.x.ai`, prefix `xai/`) is not Groq.

## English

`--profile consumer` is a low-cost smoke: mock domain, 1 task, 1 trial, concurrency 1, 8 steps, Grok defaults. It prints a short summary and stops with a clear message when `XAI_API_KEY` is missing. Explicit flags such as `--domain` or `--max-steps` still win.

```bash
tau2 run --profile consumer
```

`--profile enterprise --enterprise-config profile.yaml` reads a YAML file for per-role model routing, concurrency, a hard cost/token stop, retry backoff on each model call, and a secret-free JSONL audit log. Reaching a cap blocks the next call (`0` blocks the first one); a call that spends past the cap stops the run where it happened. The log and `edition-metadata.json` record the seed, model ids, and git sha. See [docs/examples/enterprise.profile.yaml](examples/enterprise.profile.yaml).

```bash
tau2 run --profile enterprise --enterprise-config docs/examples/enterprise.profile.yaml \
  --domain mock --num-tasks 1
```

An explicit CLI flag (`--agent-llm`, `--user-llm`, `--review-model`, `--tool-mentor-model`, `--max-concurrency`, `--seed`, `--save-to`) overrides the YAML value. Task-level `--max-retries` is unchanged. The YAML `retry` block wraps each model call.

## 한국어 요약

`--profile`을 빼면 지금의 기본값이 그대로입니다. 텍스트 역할은 `xai/grok-4.7`, 도구 멘토는 `xai/grok-4.20-0309-non-reasoning`입니다. xAI Grok(`api.x.ai`, 접두사 `xai/`)는 Groq가 아닙니다.

`--profile consumer`는 낮은 비용의 스모크 평가입니다. mock 도메인, 과제 1개, 시도 1회, 동시 실행 1, 최대 8단계이고 모델은 Grok 기본값입니다. `XAI_API_KEY`가 없으면 호출 전에 안내하고 멈춥니다. `--domain`, `--max-steps`처럼 직접 준 플래그가 우선합니다.

```bash
tau2 run --profile consumer
```

`--profile enterprise --enterprise-config profile.yaml`은 YAML로 역할별 모델을 나누고, 동시 실행 수, 비용·토큰 한도(한도에 닿으면 다음 호출을 막고, 넘긴 호출은 그 자리에서 실행을 멈춤), 모델 호출마다 재시도/백오프, 비밀이 없는 JSONL 감사 기록을 켭니다. 시드, 모델 id, git sha는 `edition-metadata.json`과 로그에 남습니다. 예시는 [docs/examples/enterprise.profile.yaml](examples/enterprise.profile.yaml)입니다.

```bash
tau2 run --profile enterprise --enterprise-config docs/examples/enterprise.profile.yaml \
  --domain mock --num-tasks 1
```

`--agent-llm`, `--user-llm`, `--review-model`, `--tool-mentor-model`, `--max-concurrency`, `--seed`, `--save-to`를 직접 주면 YAML보다 그 값이 우선합니다. 과제 단위 `--max-retries`는 그대로이고, YAML의 `retry`는 각 모델 호출에만 적용됩니다.
