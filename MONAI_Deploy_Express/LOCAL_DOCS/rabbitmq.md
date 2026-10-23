# RabbitMQ

Container `mdl-rabbitmq`, vhost `monaideploy`, UI http://127.0.0.1:15672 (admin / admin), data `.md/rabbitmq/`.

It is the message bus between IG, WM and TM, **not a work queue**: each service consumes the moment a
message arrives, so queues normally show 0 waiting. To see what is pending or running, look at WM runs
(LOCAL_SETUP "What ran"), not here.

What to look at:

- `md.workflow.request`: IG → WM, one per payload.
- `md.tasks.dispatch` / `md.tasks.update`: WM ↔ TM.
- `md.export.request.monaiscu` / `md.export.complete`: WM ↔ IG exports.
- `*-dead-letter`: messages a consumer rejected. Anything sitting here is worth reading.

```bash
docker exec mdl-rabbitmq rabbitmqctl list_queues -p monaideploy name messages consumers
docker exec mdl-rabbitmq rabbitmqadmin -u admin -p admin -V monaideploy get queue=<queue> ackmode=reject_requeue_true count=1
docker exec mdl-rabbitmq rabbitmqctl purge_queue -p monaideploy <queue>
```
