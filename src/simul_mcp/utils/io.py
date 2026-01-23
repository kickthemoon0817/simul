"""
I/O utilities for Isaac Sim MCP Server.

This module provides safe file operations, directory management, and file validation
utilities with proper error handling and security considerations.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Union, Any, Dict
import json
import yaml
from contextlib import contextmanager

from ..logging import get_logger

logger = get_logger(__name__)


def ensure_directory(path: Union[str, Path], mode: int = 0o755) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure
        mode: Directory permissions (default: 0o755)

    Returns:
        Path object for the directory

    Raises:
        OSError: If directory cannot be created
    """
    path = Path(path)

    try:
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        logger.debug(f"Ensured directory exists: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise OSError(f"Cannot create directory {path}: {e}")


def safe_file_read(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
    max_size_mb: Optional[int] = None,
) -> str:
    """
    Safely read a text file with size and encoding validation.

    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)
        max_size_mb: Maximum file size in MB (None for no limit)

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is too large
        UnicodeDecodeError: If file cannot be decoded
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Check file size if limit specified
    if max_size_mb is not None:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB")

    try:
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        logger.debug(f"Successfully read file: {file_path}")
        return content
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode file {file_path} with encoding {encoding}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise


def safe_file_write(
    file_path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
    backup: bool = True,
    atomic: bool = True,
) -> None:
    """
    Safely write content to a file with backup and atomic write options.

    Args:
        file_path: Path to the file to write
        content: Content to write
        encoding: File encoding (default: utf-8)
        backup: Create backup of existing file (default: True)
        atomic: Use atomic write (write to temp file then rename) (default: True)

    Raises:
        OSError: If file cannot be written
    """
    file_path = Path(file_path)

    # Ensure parent directory exists
    ensure_directory(file_path.parent)

    # Create backup if file exists and backup is requested
    if backup and file_path.exists():
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        try:
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup {backup_path}: {e}")

    try:
        if atomic:
            # Atomic write: write to temp file then rename
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding=encoding,
                dir=file_path.parent,
                prefix=f".{file_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)

            # Atomic rename
            temp_path.replace(file_path)
            logger.debug(f"Atomically wrote file: {file_path}")
        else:
            # Direct write
            with open(file_path, "w", encoding=encoding) as f:
                f.write(content)
            logger.debug(f"Wrote file: {file_path}")

    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        raise OSError(f"Cannot write file {file_path}: {e}")


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return file_path.stat().st_size


def is_file_readable(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is readable.

    Args:
        file_path: Path to the file

    Returns:
        True if file is readable, False otherwise
    """
    file_path = Path(file_path)

    try:
        return (
            file_path.exists() and file_path.is_file() and os.access(file_path, os.R_OK)
        )
    except Exception:
        return False


def validate_file_extension(
    file_path: Union[str, Path],
    allowed_extensions: List[str],
    case_sensitive: bool = False,
) -> bool:
    """
    Validate file extension against allowed extensions.

    Args:
        file_path: Path to the file
        allowed_extensions: List of allowed extensions (with or without dots)
        case_sensitive: Whether comparison is case sensitive

    Returns:
        True if extension is allowed, False otherwise
    """
    file_path = Path(file_path)
    file_ext = file_path.suffix

    if not case_sensitive:
        file_ext = file_ext.lower()
        allowed_extensions = [ext.lower() for ext in allowed_extensions]

    # Normalize extensions (ensure they start with dot)
    normalized_extensions = []
    for ext in allowed_extensions:
        if not ext.startswith("."):
            ext = "." + ext
        normalized_extensions.append(ext)

    return file_ext in normalized_extensions


@contextmanager
def create_temp_file(
    suffix: str = "",
    prefix: str = "simul_mcp_",
    dir: Optional[Union[str, Path]] = None,
    text: bool = True,
    delete: bool = True,
):
    """
    Context manager for creating temporary files.

    Args:
        suffix: File suffix
        prefix: File prefix
        dir: Directory for temp file (None for system temp)
        text: Open in text mode
        delete: Delete file when context exits

    Yields:
        Temporary file object
    """
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=prefix,
            dir=dir,
            mode="w+t" if text else "w+b",
            delete=delete,
        )
        logger.debug(f"Created temporary file: {temp_file.name}")
        yield temp_file
    finally:
        if temp_file and not temp_file.closed:
            temp_file.close()
            if delete and Path(temp_file.name).exists():
                try:
                    os.unlink(temp_file.name)
                    logger.debug(f"Deleted temporary file: {temp_file.name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete temporary file {temp_file.name}: {e}"
                    )


def cleanup_temp_files(
    pattern: str = "simul_mcp_*", temp_dir: Optional[Union[str, Path]] = None
) -> int:
    """
    Clean up temporary files matching a pattern.

    Args:
        pattern: File pattern to match (glob pattern)
        temp_dir: Temporary directory to clean (None for system temp)

    Returns:
        Number of files cleaned up
    """
    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir())
    else:
        temp_dir = Path(temp_dir)

    if not temp_dir.exists():
        return 0

    cleaned_count = 0
    try:
        for temp_file in temp_dir.glob(pattern):
            if temp_file.is_file():
                try:
                    temp_file.unlink()
                    cleaned_count += 1
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {temp_file}: {e}")
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {e}")

    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} temporary files")

    return cleaned_count


def load_json_file(
    file_path: Union[str, Path], max_size_mb: Optional[int] = None
) -> Dict[str, Any]:
    """
    Load and parse a JSON file.

    Args:
        file_path: Path to JSON file
        max_size_mb: Maximum file size in MB

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    content = safe_file_read(file_path, max_size_mb=max_size_mb)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise


def save_json_file(
    file_path: Union[str, Path],
    data: Dict[str, Any],
    indent: int = 2,
    backup: bool = True,
) -> None:
    """
    Save data to a JSON file.

    Args:
        file_path: Path to JSON file
        data: Data to save
        indent: JSON indentation
        backup: Create backup of existing file
    """
    content = json.dumps(data, indent=indent, ensure_ascii=False)
    safe_file_write(file_path, content, backup=backup)


def load_yaml_file(
    file_path: Union[str, Path], max_size_mb: Optional[int] = None
) -> Dict[str, Any]:
    """
    Load and parse a YAML file.

    Args:
        file_path: Path to YAML file
        max_size_mb: Maximum file size in MB

    Returns:
        Parsed YAML data

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    content = safe_file_read(file_path, max_size_mb=max_size_mb)
    try:
        return yaml.safe_load(content) or {}
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in file {file_path}: {e}")
        raise


def save_yaml_file(
    file_path: Union[str, Path], data: Dict[str, Any], backup: bool = True
) -> None:
    """
    Save data to a YAML file.

    Args:
        file_path: Path to YAML file
        data: Data to save
        backup: Create backup of existing file
    """
    content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
    safe_file_write(file_path, content, backup=backup)
