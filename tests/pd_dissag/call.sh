curl -X POST -s http://localhost:10001/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "base_model",
"prompt": "Introduce Bruce Lee in details",
"max_tokens": 100,
"temperature": 0
}'
