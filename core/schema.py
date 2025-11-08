import abc
import dataclasses


@dataclasses.dataclass
class Config(abc.ABC):
    """Base class for all configurations (algorithms, trainers, reasoning modes, etc.).

    Subclasses should be simple dataclasses with JSON-serializable fields.
    """

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Config":
        try:
            return cls(**payload)
        except Exception:
            # Fallback to default ctor if payload incompatible
            return cls()  # type: ignore[misc]

    @classmethod
    def from_env(cls) -> "Config":
        """Optional: construct config from environment variables.
        Default does nothing; subclasses override if needed.
        """
        return cls()  # type: ignore[misc]

    @abc.abstractmethod
    def identifier(self, **kwargs) -> str:
        """Return a string identifier for the config."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _is_valid_identifier(cls, identifier: str) -> bool:
        """Check if the identifier is a valid Config identifier."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _parse_identifier(cls, identifier: str, **kwargs) -> "Config":
        """Parse an Config from its identifier string or a longer string containing it."""
        raise NotImplementedError

    @classmethod
    def from_identifier(cls, identifier: str, **kwargs) -> "Config":
        """Parse an Config from its identifier string or a longer string containing it.

        Format: red_team_mode-few_shot_strategy-s{0/1}-e{n}-b{n}-p{0/1}
        Example: rbr-p-s1-e1-b2-p0
        Can also handle: QualitativeJudge_Model_rbr-p-s1-e1-b2-p0
        """
        # If the identifier contains underscores or slashes, it might be a longer string, possibly a path
        if "/" in identifier:
            parts_by_slash = identifier.split("/")
            for part in parts_by_slash[::-1]:
                try:
                    config = cls.from_identifier(part, **kwargs)
                    assert config is not None, f"Failed to parse config from {part}"
                    return config
                except ValueError:
                    continue
            raise ValueError(f"No valid config identifier found in: {identifier}")

        # Extract the actual identifier part
        if "_" in identifier:
            # Handle cases like QualitativeJudge_Model_identifier
            # First handle double underscore for comparison cases
            main_part = identifier.split("__")[0] if "__" in identifier else identifier

            # Split by underscore and find the identifier
            parts_by_underscore = main_part.split("_")

            # Search backwards for a valid identifier pattern (at least 6 hyphen-separated parts)
            actual_identifier = None
            for i in range(len(parts_by_underscore) - 1, -1, -1):
                candidate = parts_by_underscore[i]
                if cls._is_valid_identifier(candidate):
                    actual_identifier = candidate
                    break

            if actual_identifier:
                identifier = actual_identifier
            else:
                # No valid identifier found in the string
                raise ValueError(f"No valid config identifier found in: {identifier}")

        return cls._parse_identifier(identifier, **kwargs)
