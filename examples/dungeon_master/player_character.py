from enum import Enum
from random import randrange
from typing import Annotated, Self

from pydantic import BaseModel, Field


class CharacterClass(str, Enum):
    WIZARD = "WIZARD"
    WARRIOR = "WARRIOR"
    ROGUE = "ROGUE"


class NameAndCharacterClass(BaseModel):
    name: str
    """The character's name"""

    character_class: CharacterClass
    """The character's class"""


class AbilityScores(BaseModel):
    STR: Annotated[int, Field(strict=True, ge=0, le=10)]
    DEX: Annotated[int, Field(strict=True, ge=0, le=10)]
    INT: Annotated[int, Field(strict=True, ge=0, le=10)]
    WIS: Annotated[int, Field(strict=True, ge=0, le=10)]
    WIL: Annotated[int, Field(strict=True, ge=0, le=10)]
    CHA: Annotated[int, Field(strict=True, ge=0, le=10)]

    @classmethod
    def get_random_scores(cls) -> Self:
        return cls(
            STR=int(randrange(0, 3)),
            DEX=int(randrange(0, 3)),
            INT=int(randrange(0, 3)),
            WIS=int(randrange(0, 3)),
            WIL=int(randrange(0, 3)),
            CHA=int(randrange(0, 3)),
        )


class InventoryItem(BaseModel):
    slots: Annotated[int, Field(strict=True, ge=1, le=3)]
    """The number of inventory slots this item consumes. Items that can easily be carried in 1-hand (e.g. a longsword) consume 1 slot. Items that require 2 hands to carry (e.g. a haldberd) consume 2 slots. Especially bulky items consume 3 slots. An item may not occupy more than 3 slots"""

    name: str
    """The name of the item (e.g. Rusty Longsword)"""

    description: str
    """A description of the item, including any unique characteristics it has, special abilities it grants or spells it contains."""


class InventoryItems(BaseModel):
    items: list[InventoryItem]
    """The player's inventory items"""


class PlayerCharacter(BaseModel):
    """The player character for the game."""

    name: str
    """The character's name."""

    character_class: CharacterClass
    """The character's class"""

    abillity_scores: AbilityScores

    level: Annotated[int, Field(ge=1, le=10)] = 1
    """The character's current level"""

    iventory_items: Annotated[list[InventoryItem], Field(max_length=10)] = Field(default_factory=list)
    """The inventory items a palyer is carrying"""

    hp: int
    """The player's total HP. If they reach 0 HP, they die immediately."""
