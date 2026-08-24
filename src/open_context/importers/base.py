"""The importer interface.

```
source stream -> ConversationImporter -> RawEvent | MalformedRecord -> archive
```

An importer knows one provider's transport format and nothing else. It does not
open files, choose session ids, write to the archive, decide what to do about a
broken record, or count anything. Those are the same for every provider and live
in the pipeline, so a new adapter is only ever the part that is genuinely
provider-specific.

**A read yields a union.** ``read`` yields ``RawEvent`` for a record it could
normalize and ``MalformedRecord`` for one it could not. It does not raise on a
bad record, for two reasons. A generator that raises is finished, so a caller
that wanted to note the problem and continue could not; and returning the
failure as a value makes "do not silently drop data" a property of the type
rather than a rule everyone has to remember. Refusing the whole import is a
policy the pipeline applies to a yielded failure, not something the adapter
decides on the caller's behalf.

**Detection is a sample, not a parse.** ``detect`` sees the opening characters of
a source and answers whether this adapter should be the one to read it. It must
not consume the stream and must not be expensive.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, TextIO, runtime_checkable

from open_context.importers.events import RawEvent
from open_context.importers.report import MalformedRecord

ReadResult = RawEvent | MalformedRecord
"""What one source record becomes: an event, or a located explanation of why not."""


@runtime_checkable
class ConversationImporter(Protocol):
    """Translates one provider's conversation format into normalized events."""

    @property
    def provider(self) -> str:
        """Stable name for this source format, recorded on every event."""

    def detect(self, sample: str) -> bool:
        """Whether this importer should read a source starting with ``sample``."""

    def read(self, stream: TextIO) -> Iterator[ReadResult]:
        """Yield one result per source record, in source order.

        Must stream. A source larger than memory has to import, so an
        implementation that reads the whole stream before yielding is a defect
        even when its output is correct.
        """


__all__ = ["ConversationImporter", "ReadResult"]
