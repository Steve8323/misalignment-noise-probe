#!/bin/bash
PODID="9o4rxt3juykk4s"
HOOK="https://webhook.site/e55d3fcb-d2b7-4dcf-951c-be9a4dd5718b"
KEY=$(cat /workspace/.rpkey)
cd /workspace
for i in $(seq 1 480); do
  [ -f /workspace/DONE ] && break
  sleep 15
done
RES_URL="none"
[ -f /workspace/align_judge_results.json ] && RES_URL=$(curl -s --data-binary @/workspace/align_judge_results.json https://paste.rs)
LOG_URL=$(curl -s --data-binary @/workspace/run.log https://paste.rs)
STATUS="JUDGE_FAILED"
grep -q "saved align_judge_results.json" /workspace/run.log && STATUS="JUDGE_DONE"
PAYLOAD="{\"status\":\"$STATUS\",\"results\":\"$RES_URL\",\"log\":\"$LOG_URL\",\"pod\":\"$PODID\"}"
for k in 1 2 3; do
  curl -s -X POST "$HOOK" -H "Content-Type: application/json" -d "$PAYLOAD" >/dev/null
  sleep 2
done
curl -s -X DELETE "https://rest.runpod.io/v1/pods/$PODID" -H "Authorization: Bearer $KEY" >/dev/null
sleep 3
curl -s -X POST "https://api.runpod.io/graphql?api_key=$KEY" -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation{ podTerminate(input:{podId:\\\"$PODID\\\"}) }\"}" >/dev/null
