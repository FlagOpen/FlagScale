import datetime
import json
import threading
import time
from typing import Any, Callable, Dict, Optional, Union, List

from redis import ConnectionPool, Redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError, WatchError


class Communicator:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        clear: bool = False,
        password: Optional[str] = None,
    ):
        print(f"Connecting to Redis at {host}:{port}, db: {db}, password: {password}")
        self.pool = ConnectionPool(
            host=host, port=port, db=db, password=password, decode_responses=True
        )

        if clear == True:
            redis_client = Redis(connection_pool=self.pool)
            redis_client.flushdb()
            print("Database cleared.")

    def _get_conn(self) -> Redis:
        """Get a Redis connection from the pool."""
        return Redis(connection_pool=self.pool)

    # ----------------- send/recive -----------------
    def send(self, channel: str, message: str) -> bool:
        """send a message to a Redis channel.
        Returns True if the message was published successfully, False otherwise.
        """
        try:
            redis_client = self._get_conn()
            return redis_client.publish(channel, message) > 0
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while publishing to Redis: {e}")
        finally:
            redis_client.close()

    def listen(
        self,
        channel: str,
        callback: Callable[[Dict[str, Any]], None],
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Subscribe to a Redis channel and call the callback function with the message.
        The callback function should accept a single argument, which is the message.
        If stop_event is provided, the subscription will stop when the event is set.
        """

        conn = self._get_conn()
        pubsub = conn.pubsub()
        pubsub.subscribe(channel)
        for message in pubsub.listen():
            print(f"Received message: {message}, {channel}, {datetime.datetime.now()}")
            if stop_event and stop_event.is_set():
                break
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    callback(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON message: {e}")

    # ----------------- data -----------------
    def record(self, name: str, key: str, value: str) -> None:
        """Set a field in a hash."""
        try:
            redis_client = self._get_conn()
            return redis_client.hset(name, key, value) == 1
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while setting hash in Redis: {e}")
            return False

    def read(self, name: str, key: str) -> Optional[Dict[str, Any]]:
        """Get a field from a hash."""
        try:
            redis_client = self._get_conn()
            data = redis_client.hget(name, key)
            return json.loads(data) if data else None
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while getting hash from Redis: {e}")
            return None

    def read_all(self, name: str) -> Optional[Dict[str, Any]]:
        """Get all fields from a hash."""
        try:
            redis_client = self._get_conn()
            data = redis_client.hgetall(name)
            return {k: json.loads(v) for k, v in data.items()}
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while getting all hashes from Redis: {e}")
            return {}

    def register(
        self, name: str, value: str, expire_second: Optional[int] = None
    ) -> bool:
        """Set a key in Redis."""
        try:
            redis_client = self._get_conn()
            return redis_client.set(name, value, ex=expire_second) == True
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while setting key in Redis: {e}")
            return False

    def retrieve(
        self, key: str, deserialize_json: bool = True
    ) -> Optional[Union[str, dict, list]]:
        try:
            redis_client = self._get_conn()
            value = redis_client.get(key)
            if value is None:
                return None

            if deserialize_json:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value.decode("utf-8") if isinstance(value, bytes) else value
            else:
                return value.decode("utf-8") if isinstance(value, bytes) else value
        except (ConnectionError, TimeoutError, RedisError) as e:
            return None

    def set_ttl(
        self,
        key: str,
        seconds: int,
    ) -> bool:
        redis_client = self._get_conn()
        return bool(redis_client.expire(key, seconds))

    def get_all_keys(self, pattern: str) -> list:
        """Get all keys starting with the given pattern in Redis."""
        try:
            redis_client = self._get_conn()
            keys = list(redis_client.scan_iter(pattern))
            return keys
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while getting keys from Redis: {e}")
            return []

    def gat_all_values(self, pattern: str) -> List:
        """Get all keys and values starting with the given pattern in Redis."""
        try:
            redis_client = self._get_conn()
            keys = redis_client.scan_iter(pattern)
            if not keys:
                return []
            values = redis_client.mget(keys)
            return [json.loads(v) for v in values if v is not None]

        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while getting keys and values from Redis: {e}")
            return {}
    
    def wait_for_all_channels_response(
        self, channels: list[str], task_id: str, timeout: int = 6000
    ) -> Dict[str, Optional[str]]:
        """Listen to multiple Redis channels until all receive specified task_id or timeout.

        Continuously monitors the given Redis channels until either:
        1. All channels have received a message containing the specified task_id, OR
        2. The specified timeout period is reached

        Parameters
        ----------
        redis_client : Redis
            Redis connection object
        channels : list[str]
            List of channel names to subscribe to
        task_id : str
            The task ID to match in incoming messages
        timeout : float
            Maximum waiting time in seconds

        Returns
        -------
        dict[str, Optional[str]]
            Dictionary containing:
            - Channel names as keys
            - Received message (str) if matching task_id was found, else None
            - Special key "__timeout__" (bool) indicating if timeout occurred

        Examples
        --------
        >>> result = wait_for_redis_messages(
        ...     redis_client=redis_conn,
        ...     channels=["channel1", "channel2"],
        ...     task_id="task_123",
        ...     timeout=10.0
        ... )
        >>> print(result)
        {
            "channel1": "task_123:done",
            "channel2": None,
            "__timeout__": True
        }
        """
        redis_client = self._get_conn()
        pubsub = redis_client.pubsub()
        pubsub.subscribe(*channels)

        result = {channel: None for channel in channels}
        start_time = time.time()

        while time.time() - start_time < timeout:
            message = pubsub.get_message()
            if message and message["type"] == "message":
                channel = message["channel"]
                msg_data = json.loads(message["data"])
                if msg_data.get("task_id") == task_id:
                    result[channel] = msg_data

                    if all(v is not None for v in result.values()):
                        pubsub.unsubscribe()
                        return result
            time.sleep(0.1)
        pubsub.unsubscribe()
        result["__timeout__"] = True
        return result

    def update_json_field_py(self, key: str, field_path: str, new_value: Any) -> bool:
        redis_client = self._get_conn()
        pipe = redis_client.pipeline()

        try:
            while True:
                try:
                    pipe.watch(key)
                    json_str = pipe.get(key)
                    if not json_str:
                        pipe.unwatch()
                        return False
                    data = json.loads(json_str)
                    data[field_path] = new_value
                    pipe.multi()
                    pipe.set(key, json.dumps(data))
                    pipe.execute()
                    return True

                except WatchError:
                    continue
        except RedisError as e:
            return False

    # ----------------- Close Connection -----------------
    def close(self) -> None:
        """Close the Redis connection pool."""
        self.pool.disconnect()
        print("Redis connection pool closed.")
