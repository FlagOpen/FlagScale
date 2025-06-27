import datetime
import threading
import time

from typing import Any, Callable, Dict, List, Optional, Union

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
        """
        Initialize Redis communicator with individual parameters.

        Args:
            host (str): Redis server hostname/IP. Default: "localhost".
            port (int): Redis server port. Default: 6379.
            db (int): Redis database index. Default: 0.
            clear (bool): If True, flushes the database on initialization. Default: False.
            password (Optional[str]): Redis authentication password. Default: None.
        """
        self.host = host
        self.port = port
        self.db = db
        self.clear = clear
        self.password = password

        # Log connection details (mask password for security)
        print(f"Connecting to Redis at {host}:{port}, db: {db}")

        # Create Redis connection pool
        self.pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,  # Automatically decode byte responses to strings
        )

        # Clear database if requested
        if clear:
            self._clear_db()

    @classmethod
    def from_config(cls, config: Dict[str, Union[str, int, bool]]) -> "Communicator":
        """
        Alternative constructor that initializes from a configuration dictionary.

        Args:
            config (Dict): Dictionary containing Redis connection parameters.
                Supported keys:
                    - host (str)
                    - port (int)
                    - db (int)
                    - password (Optional[str])
                    - clear (bool)

        Returns:
            Communicator: New instance configured with the provided settings.

        Example:
            >>> config = {
            ...     "host": "redis.example.com",
            ...     "port": 6380,
            ...     "db": 1,
            ...     "password": "secret",
            ...     "clear": True
            ... }
            >>> comm = Communicator.from_config(config)
        """
        return cls(
            host=config.get("host", "localhost"),  # Fallback to default if not provided
            port=config.get("port", 6379),
            db=config.get("db", 0),
            password=config.get("password"),  # None if not provided
            clear=config.get("clear", False),
        )

    def _clear_db(self) -> None:
        """Flushes the current Redis database."""
        with Redis(connection_pool=self.pool) as redis_client:
            redis_client.flushdb()
            print("Database cleared successfully.")

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
                callback(message["data"])

    # ----------------- data -----------------
    def record_agent_status(self, name: str, value: str, _: Optional[float] = None) -> bool:
        """Append a member to short-term status list (score parameter is ignored)."""
        try:
            redis_client = self._get_conn()
            return redis_client.rpush(f"SHORT_STATUS:{name}", value) > 0
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while appending to short-term status list: {e}")
            return False

    def read_agent_status(self, name: str) -> List[str]:
        """Get all members from short-term status list."""
        try:
            redis_client = self._get_conn()
            return redis_client.lrange(f"SHORT_STATUS:{name}", 0, -1)
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while reading short-term status list: {e}")
            return []

    def clear_agent_status(self, name: str) -> bool:
        """Delete short-term status list."""
        try:
            redis_client = self._get_conn()
            return redis_client.delete(f"SHORT_STATUS:{name}") == 1
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while clearing short-term status list: {e}")
            return False

    def register_agent(
        self, agent_name: str, agent_data: Dict[str, str], expire_second: Optional[int] = None
    ) -> bool:
        """Register agent in Redis under AGENT_INFO hash.

        Creates AGENT_INFO hash if not exists.

        Args:
            agent_name (str): Key identifier for the agent
            agent_data (Dict[str, str]): Agent attributes
            expire_second (Optional[int]): TTL in seconds for the AGENT_INFO hash

        Returns:
            bool: True if successful, False on failure
        """
        try:
            redis_client = self._get_conn()

            # Pipeline both operations atomically
            with redis_client.pipeline() as pipe:
                # 1. Store agent data in AGENT_INFO hash
                pipe.hset("AGENT_INFO", key=agent_name, value=agent_data)

                # 2. Set expiration if specified
                if expire_second is not None:
                    pipe.expire("AGENT_INFO", expire_second)

                pipe.execute()

            self.send("AGENT_REGISTRATION", agent_name)

            return True

        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Failed to register agent {agent_name}: {e}")
            return False

    def retrieve_agent(self, agent_name: str) -> Optional[Dict[str, str]]:
        """Retrieve agent data from AGENT_INFO hash."""
        try:
            redis_client = self._get_conn()
            return redis_client.hget("AGENT_INFO", agent_name)
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error retrieving agent {agent_name}: {e}")
            return None

    def retrieve_all_agents(self) -> Dict[str, Dict[str, str]]:
        """Retrieve all agents from AGENT_INFO hash."""
        try:
            redis_client = self._get_conn()
            return redis_client.hgetall("AGENT_INFO")
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error retrieving agent registry: {e}")
            return {}

    def retrieve_all_agents_name(self) -> List[str]:
        """Retrieve all agent names (keys) from AGENT_INFO hash.

        Returns:
            List[str]: List of all agent names/keys.
            Returns empty list if no agents exist or error occurs.
        """
        try:
            redis_client = self._get_conn()
            return list(redis_client.hkeys("AGENT_INFO"))

        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error retrieving agent names: {e}")
            return []

    def agent_heartbeat(self, agent_name: str, seconds: int) -> bool:
        """Set TTL for the agent's registration in AGENT_INFO hash.

        Args:
            agent_name: Name of the registered agent
            seconds: TTL in seconds (must be > 0)

        Returns:
            bool: True if TTL was set successfully, False otherwise
        """
        try:
            redis_client = self._get_conn()

            # Verify agent exists
            if not redis_client.hexists("AGENT_INFO", agent_name):
                return False

            # Set TTL for the entire hash
            return bool(redis_client.expire("AGENT_INFO", seconds))

        except (ConnectionError, TimeoutError, RedisError):
            return False

    def update_agent_busy(self, agent_name: str, busy: bool) -> bool:
        """Update agent's busy status in the AGENT_BUSY hash.

        Args:
            agent_name (str): Name identifier for the agent
            busy (bool): True for busy, False for available

        Returns:
            bool: True if update succeeded, False on failure

        Example:
            >>> comm.update_agent_busy("robot_1", True)  # Set busy
            >>> comm.update_agent_busy("robot_1", False) # Set available
        """
        try:
            redis_client = self._get_conn()
            return redis_client.hset("AGENT_BUSY", agent_name, int(busy)) >= 0
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error updating busy status for {agent_name}: {e}")
            return False

    def agent_is_busy(self, agent_name: str) -> Optional[bool]:
        """Get current busy status of an agent.

        Args:
            agent_name (str): Agent name to query

        Returns:
            Optional[bool]:
                - True if agent is busy
                - False if available
                - None if record not found or error occurred
        """
        try:
            redis_client = self._get_conn()
            status = redis_client.hget("AGENT_BUSY", agent_name)
            return bool(int(status)) if status is not None else None
        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error getting busy status for {agent_name}: {e}")
            return None

    def wait_agents_free(
        self, agents_name: list[str], check_interval: float = 0.5, timeout: Optional[float] = None
    ) -> bool:
        """Wait until all specified agents become free (busy=False).

        Args:
            agents_name: List of agent names to monitor
            check_interval: Seconds between status checks (default: 0.5)
            timeout: Maximum wait time in seconds (None = no timeout)

        Returns:
            bool:
                - True if all agents became free
                - False if timeout occurred

        Example:
            >>> # Wait for robot1 and robot2 to become free
            >>> success = comm.wait_agent_free(["robot1", "robot2"])
            >>> if success:
            >>>     print("All agents are now available")
        """
        start_time = time.time()

        try:
            redis_client = self._get_conn()

            while True:
                # Check timeout
                if timeout is not None and (time.time() - start_time) > timeout:
                    return False

                # Get all statuses in one atomic operation
                statuses = redis_client.hmget("AGENT_BUSY", agents_name)

                # Check if all are free (None means no record = considered free)
                all_free = True
                for status in statuses:
                    if status is not None and bool(int(status)):
                        all_free = False
                        break

                if all_free:
                    return True

                # Wait before next check
                time.sleep(check_interval)

        except (ConnectionError, TimeoutError, RedisError) as e:
            print(f"Error while waiting for agent status: {e}")
            return False

    # ----------------- Close Connection -----------------
    def _close_db(self) -> None:
        """Close the Redis connection pool."""
        self.pool.disconnect()
        print("Redis connection pool closed.")
