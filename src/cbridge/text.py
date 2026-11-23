"""Documents and the byte counts every metric is normalised by."""

from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    source: str
    date: str  # ISO date the text was spoken or published
    title: str
    turns: list[tuple[str, str]]  # (speaker, what they said)
    meta: dict = field(default_factory=dict)

    def render(self) -> tuple[str, list[tuple[int, int]]]:
        """The text every model sees, and the character spans of speaker labels (scored, but also reported apart)."""
        parts, labels, pos = [], [], 0
        for speaker, said in self.turns:
            label = f"{speaker}: " if speaker else ""
            labels.append((pos, pos + len(label)))
            chunk = label + said.strip() + "\n\n"
            parts.append(chunk)
            pos += len(chunk)
        return "".join(parts).rstrip("\n") + "\n", labels

    def to_json(self) -> dict:
        return dict(id=self.id, source=self.source, date=self.date, title=self.title,
                    turns=[list(t) for t in self.turns], meta=self.meta)

    @classmethod
    def from_json(cls, d: dict) -> "Document":
        return cls(d["id"], d["source"], d["date"], d["title"], [tuple(t) for t in d["turns"]], d.get("meta", {}))


def utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))
