"""Estado de los jobs y cola de trabajo sobre Redis (cliente async `redis.asyncio`)."""

import json
import logging
from typing import Any

from pydantic import TypeAdapter, ValidationError
from redis.asyncio import Redis

from app.schemas import JobStatus, QueueMessage, RunMessage, TaskRequest

log = logging.getLogger(__name__)

_queue_adapter: TypeAdapter[QueueMessage] = TypeAdapter(QueueMessage)

# Compare-and-set atómico: evita que dos aprobaciones simultáneas reanuden el mismo job.
_TRANSITION = """
if redis.call('HGET', KEYS[1], 'status') == ARGV[1] then
  redis.call('HSET', KEYS[1], 'status', ARGV[2])
  return 1
end
return 0
"""


def job_key(job_id: str) -> str:
    return f"task:{job_id}"


class JobStore:
    def __init__(self, client: Redis, queue_name: str) -> None:
        self._r = client
        self._queue = queue_name

    @classmethod
    def from_url(cls, redis_url: str, queue_name: str) -> "JobStore":
        return cls(Redis.from_url(redis_url, decode_responses=True), queue_name)

    async def ping(self) -> bool:
        try:
            return bool(await self._r.ping())
        except Exception:  # noqa: BLE001
            return False

    async def create_job(self, job_id: str, task: TaskRequest) -> None:
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.hset(job_key(job_id), mapping={"status": JobStatus.PENDING, "task": task.model_dump_json()})
            pipe.rpush(self._queue, RunMessage(job_id=job_id, job_data=task).model_dump_json())
            await pipe.execute()

    async def get(self, job_id: str) -> dict[str, str] | None:
        return await self._r.hgetall(job_key(job_id)) or None

    async def update(self, job_id: str, **fields: Any) -> None:
        mapping = {k: v if isinstance(v, str) else json.dumps(v) for k, v in fields.items() if v is not None}
        await self._r.hset(job_key(job_id), mapping=mapping)

    async def transition(self, job_id: str, expected: JobStatus, new: JobStatus) -> bool:
        return bool(await self._r.eval(_TRANSITION, 1, job_key(job_id), expected.value, new.value))

    async def enqueue(self, message: QueueMessage) -> None:
        await self._r.rpush(self._queue, message.model_dump_json())

    async def pop(self, timeout: int = 5) -> QueueMessage | None:
        item = await self._r.blpop([self._queue], timeout=timeout)
        if item is None:
            return None
        try:
            return _queue_adapter.validate_json(item[1])
        except ValidationError:
            log.exception("Mensaje inválido en la cola, se descarta: %s", item[1])
            return None

    async def aclose(self) -> None:
        await self._r.aclose()
