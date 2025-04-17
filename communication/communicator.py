from redis import Redis, ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError, RedisError
from typing import Optional, Dict, Any, Callable
import json
import threading

class Communicator:
    def __init__(
        self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None
    ):
        print(f"Connecting to Redis at {host}:{port}, db: {db}, password: {password}")
        self.pool = ConnectionPool(host=host, port=port, db=db, password=password, decode_responses=True)
        
    def _get_conn(self) -> Redis:
        """Get a Redis connection from the pool."""
        return Redis(connection_pool=self.pool)
    
    # ----------------- Pub/Sub -----------------
    def publish(self, channel: str, message: str) -> bool:
        """Publish a message to a Redis channel.
        Returns True if the message was published successfully, False otherwise.
        """
        try:
            redis_client = self._get_conn()
            return redis_client.publish(channel, message) > 0
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while publishing to Redis: {e}")
        finally:
            redis_client.close()
    
    def subscribe(self, channel: str,
                  callback: Callable[[Dict[str, Any]], None],
                  stop_event: Optional[threading.Event] = None) -> None:
        """Subscribe to a Redis channel and call the callback function with the message.
        The callback function should accept a single argument, which is the message.
        If stop_event is provided, the subscription will stop when the event is set.
        """
        
        conn = self._get_conn()
        pubsub = conn.pubsub()
        pubsub.subscribe(channel)
        for message in pubsub.listen():
            print(f"Received message: {message}, {channel}")
            if stop_event and stop_event.is_set():
                break
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    callback(data)    
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON message: {e}")


    # ----------------- Hashes -----------------
    def hset(self, name: str, key: str, value: str) -> None:
        """Set a field in a hash."""
        try:
            redis_client = self._get_conn()
            return redis_client.hset(name, key, value) == 1
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while setting hash in Redis: {e}")
            return False
    
    def hget(self, name: str, key: str) -> Optional[Dict[str, Any]]:
        """Get a field from a hash."""
        try:
            redis_client = self._get_conn()
            data = redis_client.hget(name, key)
            return json.loads(data) if data else None
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while getting hash from Redis: {e}")
            return None
        
    def hgetall(self, name: str) -> Optional[Dict[str, Any]]:
        """Get all fields from a hash."""
        try:
            redis_client = self._get_conn()
            data = redis_client.hgetall(name)
            return {k: json.loads(v) for k, v in data.items()}
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while getting all hashes from Redis: {e}")
            return {}
        
        
    # ----------------- Keys -----------------
    def set(self, name: str, value: str, ex: Optional[int] = None) -> bool:
        """Set a key in Redis."""
        try:
            redis_client = self._get_conn()
            return redis_client.set(name, value, ex=ex) == True
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while setting key in Redis: {e}")
            return False
        
    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a key from Redis."""
        try:
            redis_client = self._get_conn()
            data = redis_client.get(name)
            return json.loads(data) if data else None
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while getting key from Redis: {e}")
            return None
        
    # ----------------- Close Connection -----------------
    def close(self) -> None:
        """Close the Redis connection pool."""
        self.pool.disconnect()
        print("Redis connection pool closed.")
        
    def get_all_robot_keys(self, pattern: str = "robot_*") -> list:
        """Get all keys starting with the given pattern in Redis."""
        try:
            redis_client = self._get_conn()
            keys = list(redis_client.scan_iter(pattern))
            return keys
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while getting keys from Redis: {e}")
            return []
        