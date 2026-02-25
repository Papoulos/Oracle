from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union

class CreationGuide(BaseModel):
    steps: List[str]
    rules_summary: Dict
    internal_notes: str

class GardeResponse(BaseModel):
    possible: bool = Field(description="Whether the action is possible")
    raison: str = Field(description="Explanation in French. Must be 'OUI' or 'NON, parce que...'")

class RulesAnalyse(BaseModel):
    besoin_jet: bool = Field(description="Whether a dice roll is needed")
    jet_format: Optional[str] = Field(None, description="Dice expression like '1d20+2'")
    explication_regle: str = Field(description="Short explanation of the rule in French")
    seuil: Optional[Union[int, str]] = Field(None, description="Difficulty threshold")

class XPGain(BaseModel):
    xp_gagne: int = Field(default=0, description="Amount of XP gained")
    raison: str = Field(description="Reason for XP gain in French")

class LevelUpCheck(BaseModel):
    passage_niveau: bool = Field(description="Whether the character levels up")
    nouveau_niveau: int = Field(description="New level reached")

class CharacterCreationAnalysis(BaseModel):
    updates: Dict[str, Optional[Union[str, int, Dict, List]]] = Field(default_factory=dict)
    player_agreed_to_roll: bool = Field(default=False)
    missing_fields: List[str] = Field(default_factory=list, description="Fields still missing")
    stats_to_roll: List[str] = Field(default_factory=list, description="List of stats names that need a roll")
    internal_thought: str = Field(default="", description="English reasoning")

class CharacterCreationResponse(BaseModel):
    message: str = Field(description="DM response in French")
    reflexion: str = Field(description="English thought")
    creation_terminee: bool = Field(default=False)

class CharacterEvolutionResponse(BaseModel):
    reflexion: str = Field(description="English thought")
    message: str = Field(description="DM response in French")
    personnage_updates: Dict = Field(default_factory=dict)
    evolution_terminee: bool = Field(default=False)

class MemoryUpdates(BaseModel):
    personnage_updates: Dict = Field(default_factory=dict)
    monde_updates: Dict = Field(default_factory=dict)
    resume_action: str = Field(description="Short summary in French for history")
