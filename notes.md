To create custom model based on another model.

$: ollama show phi --modelfile
$: ollama show phi --modelfile > discord-modelfile
$: ollama create discord-phi --file discord-modelfile

$: ollama create openhermes-discord --file openhermes-discord-modelfile

$: docker pull redis
$: docker run -d --name my-redis-container -p 6379:6379 redis
$: docker start my-redis-container
$: docker exec -it 593d15ed477c redis-cli FLUSHALL