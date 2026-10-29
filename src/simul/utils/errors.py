"""How an exception reads in a tool's error envelope.

The MCP server's envelope uses these for every exception that escapes a tool
call, and the Blender add-on's bridge uses ``exception_message`` for the ones
raised inside Blender. The add-on bundles this module, so it imports only the
standard library.
"""

from __future__ import annotations


def exception_type_name(exc: BaseException) -> str:
    """Name an exception for an envelope's ``error_type``.

    A structured exception may declare a stable ``error_type`` string of its
    own: Blender's AttachmentError family does, and a BridgeRemoteError names
    the class the add-on raised inside Blender. Any other exception is named
    by its class.

    Args:
        exc: The exception to name.

    Returns:
        The declared type, or the exception's class name.
    """
    declared = getattr(exc, "error_type", None)
    if isinstance(declared, str) and declared:
        return declared
    return type(exc).__name__


def exception_message(exc: BaseException) -> str:
    """Describe an exception for an envelope's ``error``.

    Context attached with ``add_note`` (PEP 678) prefixes the message,
    outermost first, so a note such as ``Failed to add rigid body to 'Cube'``
    reads ``Failed to add rigid body to 'Cube': <bpy message>``. An exception
    with neither notes nor a message is named by its class rather than
    reported as an empty string.

    Args:
        exc: The exception to describe.

    Returns:
        The message with its notes' context.
    """
    notes = [
        note for note in getattr(exc, "__notes__", ()) if isinstance(note, str) and note
    ]
    parts = [*reversed(notes), str(exc)]
    return ": ".join(part for part in parts if part) or type(exc).__name__
