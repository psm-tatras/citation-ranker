# citation-ranker
A system that will identify , judge and rank citations specified in the academic papers

docker pull mongo:latest

docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=secret \
  -v mongodb_data:/Work/DockerData/Mongodata \
  mongo:latest
