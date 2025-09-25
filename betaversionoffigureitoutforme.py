import asyncio
import json
import logging
import time
import os
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessStage(Enum):
    DISCOVERY = "discovery"
    VALIDATION = "validation"  
    PLANNING = "planning"
    LAUNCH_PREP = "launch_prep"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class ExperienceLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class SkillAssessment:
    """More detailed skill assessment with proficiency levels"""
    technical_skills: Dict[str, ExperienceLevel]
    business_skills: Dict[str, ExperienceLevel]
    industry_knowledge: Dict[str, ExperienceLevel]
    soft_skills: Dict[str, ExperienceLevel]
    
    def get_skill_score(self, category: str) -> float:
        """Calculate weighted skill score for a category"""
        skill_dict = getattr(self, category, {})
        if not skill_dict:
            return 0.0
        
        level_weights = {
            ExperienceLevel.BEGINNER: 1,
            ExperienceLevel.INTERMEDIATE: 2.5,
            ExperienceLevel.ADVANCED: 4,
            ExperienceLevel.EXPERT: 5
        }
        
        total_score = sum(level_weights[level] for level in skill_dict.values())
        max_possible = len(skill_dict) * 5
        return (total_score / max_possible) * 10 if max_possible > 0 else 0

@dataclass
class MarketConditions:
    """Real-time market condition factors"""
    demand_level: float  # 1-10 scale
    competition_intensity: float  # 1-10 scale  
    market_maturity: str  # emerging/growth/mature/declining
    entry_barriers: float  # 1-10 scale
    regulatory_complexity: float  # 1-10 scale
    capital_requirements: Dict[str, float]  # min/recommended/optimal
    customer_acquisition_difficulty: float  # 1-10 scale
    
    def calculate_market_attractiveness(self) -> float:
        """Calculate overall market attractiveness score"""
        # Higher demand and lower barriers = better
        attractiveness = (
            (self.demand_level * 0.3) +
            ((10 - self.competition_intensity) * 0.2) +
            ((10 - self.entry_barriers) * 0.2) +
            ((10 - self.regulatory_complexity) * 0.15) +
            ((10 - self.customer_acquisition_difficulty) * 0.15)
        )
        return round(attractiveness, 1)

@dataclass
class EntrepreneurProfile:
    """Enhanced entrepreneurial profile with more accurate assessment"""
    # Basic Info
    name: str
    location: str
    age_range: str
    education_background: str
    work_experience: List[str]
    
    # Detailed Skills Assessment
    skills: SkillAssessment
    
    # Financial Reality Check
    available_capital: float
    monthly_personal_expenses: float
    existing_income: float  # Current income if any
    family_financial_obligations: float
    debt_obligations: float
    emergency_fund_months: int  # How many months of expenses saved
    
    # Risk & Motivation Assessment
    risk_tolerance: str  # conservative/moderate/aggressive
    motivation_level: float  # 1-10 scale
    time_available_per_week: int  # hours per week
    minimum_income_needed: float  # Minimum monthly income required
    income_timeline_need: str
    
    # Personal Constraints & Preferences
    health_limitations: List[str]
    family_commitments: str  # none/light/moderate/heavy
    travel_willingness: str  # none/local/national/international
    relocation_willingness: bool
    partnership_preference: str  # solo/small-team/larger-team
    
    # Market Position
    existing_network_strength: float  # 1-10 scale
    personal_brand_strength: float  # 1-10 scale
    credibility_factors: List[str]
    
    # Sri Lankan Context
    language_skills: Dict[str, ExperienceLevel]
    cultural_adaptability: float  # 1-10 scale
    regulatory_knowledge: ExperienceLevel
    local_market_understanding: ExperienceLevel
    
    def calculate_readiness_score(self) -> Dict[str, float]:
        """Calculate readiness across different dimensions"""
        financial_readiness = self._calculate_financial_readiness()
        skill_readiness = self._calculate_skill_readiness()
        market_readiness = self._calculate_market_readiness()
        commitment_readiness = self._calculate_commitment_readiness()
        
        overall = (financial_readiness + skill_readiness + market_readiness + commitment_readiness) / 4
        
        return {
            "financial": financial_readiness,
            "skill": skill_readiness,
            "market": market_readiness,
            "commitment": commitment_readiness,
            "overall": round(overall, 1)
        }
    
    def _calculate_financial_readiness(self) -> float:
        """Calculate financial readiness score"""
        # Emergency fund adequacy
        emergency_score = min(self.emergency_fund_months / 6, 1) * 3
        
        # Available capital relative to minimum business needs
        capital_score = min(self.available_capital / 1000000, 1) * 3  # 1M LKR baseline
        
        # Debt burden impact
        total_monthly_obligations = self.monthly_personal_expenses + self.debt_obligations + self.family_financial_obligations
        debt_burden_ratio = total_monthly_obligations / max(self.existing_income, self.minimum_income_needed)
        debt_score = max(0, (2 - debt_burden_ratio)) * 2
        
        # Income replacement capability
        income_score = (self.existing_income / max(self.minimum_income_needed, 1)) * 2
        
        return round(min(emergency_score + capital_score + debt_score + income_score, 10), 1)
    
    def _calculate_skill_readiness(self) -> float:
        """Calculate skill readiness across all categories"""
        tech_score = self.skills.get_skill_score("technical_skills") * 0.3
        business_score = self.skills.get_skill_score("business_skills") * 0.3
        industry_score = self.skills.get_skill_score("industry_knowledge") * 0.2
        soft_score = self.skills.get_skill_score("soft_skills") * 0.2
        
        return round(tech_score + business_score + industry_score + soft_score, 1)
    
    def _calculate_market_readiness(self) -> float:
        """Calculate market readiness based on local context"""
        network_score = self.existing_network_strength * 0.3
        brand_score = self.personal_brand_strength * 0.2
        
        # Language and cultural factors
        language_score = len([l for l in self.language_skills.values() if l in [ExperienceLevel.ADVANCED, ExperienceLevel.EXPERT]]) * 1.5
        cultural_score = self.cultural_adaptability * 0.2
        
        # Local market knowledge
        local_knowledge_weights = {
            ExperienceLevel.BEGINNER: 1,
            ExperienceLevel.INTERMEDIATE: 2.5,
            ExperienceLevel.ADVANCED: 4,
            ExperienceLevel.EXPERT: 5
        }
        local_score = local_knowledge_weights[self.local_market_understanding] * 0.2
        
        return round(min(network_score + brand_score + language_score + cultural_score + local_score, 10), 1)
    
    def _calculate_commitment_readiness(self) -> float:
        """Calculate commitment and motivation readiness"""
        # Time availability
        time_score = min(self.time_available_per_week / 40, 1) * 3
        
        # Motivation level
        motivation_score = self.motivation_level * 0.3
        
        # Family and personal constraints
        commitment_weights = {
            "none": 3,
            "light": 2.5,
            "moderate": 2,
            "heavy": 1
        }
        commitment_score = commitment_weights.get(self.family_commitments, 1.5)
        
        # Health considerations
        health_score = 3 if not self.health_limitations else max(1, 3 - len(self.health_limitations) * 0.5)
        
        return round(min(time_score + motivation_score + commitment_score + health_score, 10), 1)

@dataclass 
class SectorOpportunity:
    """Detailed sector opportunity analysis"""
    sector_name: str
    market_conditions: MarketConditions
    required_skills: Dict[str, ExperienceLevel]
    typical_business_models: List[Dict[str, Any]]
    success_factors: List[str]
    common_failure_reasons: List[str]
    competitive_landscape: Dict[str, Any]
    growth_trajectory: str
    regulatory_requirements: List[str]
    
    def calculate_sector_attractiveness(self) -> float:
        """Calculate overall sector attractiveness"""
        market_score = self.market_conditions.calculate_market_attractiveness()
        
        # Growth trajectory scoring
        growth_scores = {
            "declining": 2,
            "stagnant": 4,
            "stable": 6,
            "growing": 8,
            "booming": 10
        }
        growth_score = growth_scores.get(self.growth_trajectory, 5)
        
        return round((market_score * 0.7 + growth_score * 0.3), 1)

class EnhancedOpportunityAnalyzer:
    """More sophisticated opportunity analysis with better fit calculation"""
    
    def __init__(self):
        self.sectors = self._initialize_sector_data()
        
    def _initialize_sector_data(self) -> Dict[str, SectorOpportunity]:
        """Initialize comprehensive sector data"""
        return {
            "fintech": SectorOpportunity(
                sector_name="Financial Technology",
                market_conditions=MarketConditions(
                    demand_level=8.5,
                    competition_intensity=7.0,
                    market_maturity="growth",
                    entry_barriers=6.0,
                    regulatory_complexity=8.0,
                    capital_requirements={"min": 500000, "recommended": 2000000, "optimal": 10000000},
                    customer_acquisition_difficulty=7.0
                ),
                required_skills={
                    "programming": ExperienceLevel.ADVANCED,
                    "financial_analysis": ExperienceLevel.INTERMEDIATE,
                    "regulatory_compliance": ExperienceLevel.INTERMEDIATE,
                    "user_experience_design": ExperienceLevel.INTERMEDIATE,
                    "cybersecurity": ExperienceLevel.INTERMEDIATE
                },
                typical_business_models=[
                    {
                        "model": "SaaS Platform",
                        "revenue_potential": "LKR 2M-50M annually",
                        "time_to_revenue": "6-12 months",
                        "capital_intensity": "moderate"
                    },
                    {
                        "model": "Transaction Processing",
                        "revenue_potential": "LKR 5M-100M annually", 
                        "time_to_revenue": "8-18 months",
                        "capital_intensity": "high"
                    }
                ],
                success_factors=[
                    "Strong technical execution",
                    "Regulatory compliance from day one",
                    "Building user trust and credibility",
                    "Scalable technology architecture",
                    "Partnership with financial institutions"
                ],
                common_failure_reasons=[
                    "Underestimating regulatory requirements",
                    "Poor user acquisition strategy",
                    "Security vulnerabilities",
                    "Insufficient capital for compliance"
                ],
                competitive_landscape={
                    "established_players": ["Commercial Banks", "Licensed Finance Companies"],
                    "startup_competition": "High",
                    "international_threat": "Medium",
                    "differentiation_opportunities": ["Local market focus", "Mobile-first approach", "Crypto integration"]
                },
                growth_trajectory="growing",
                regulatory_requirements=[
                    "CBSL approval for payment services",
                    "AML/KYC compliance procedures",
                    "Data protection compliance",
                    "Cybersecurity standards"
                ]
            ),
            
            "import_export_tech": SectorOpportunity(
                sector_name="Import/Export Technology",
                market_conditions=MarketConditions(
                    demand_level=7.5,
                    competition_intensity=5.0,
                    market_maturity="emerging",
                    entry_barriers=7.0,
                    regulatory_complexity=9.0,
                    capital_requirements={"min": 2000000, "recommended": 10000000, "optimal": 50000000},
                    customer_acquisition_difficulty=6.0
                ),
                required_skills={
                    "supply_chain_management": ExperienceLevel.ADVANCED,
                    "international_trade": ExperienceLevel.INTERMEDIATE,
                    "logistics_software": ExperienceLevel.INTERMEDIATE,
                    "regulatory_compliance": ExperienceLevel.ADVANCED,
                    "negotiation": ExperienceLevel.INTERMEDIATE
                },
                typical_business_models=[
                    {
                        "model": "Trade Management Software",
                        "revenue_potential": "LKR 10M-200M annually",
                        "time_to_revenue": "12-24 months", 
                        "capital_intensity": "high"
                    },
                    {
                        "model": "Import/Export Consulting + Tech",
                        "revenue_potential": "LKR 5M-50M annually",
                        "time_to_revenue": "6-12 months",
                        "capital_intensity": "moderate"
                    }
                ],
                success_factors=[
                    "Deep understanding of trade regulations",
                    "Strong relationships with customs/regulatory bodies",
                    "Reliable technology infrastructure",
                    "Excellent customer service",
                    "Continuous compliance monitoring"
                ],
                common_failure_reasons=[
                    "Regulatory violations",
                    "Insufficient working capital",
                    "Poor supplier relationships",
                    "Market timing issues",
                    "Currency fluctuation losses"
                ],
                competitive_landscape={
                    "established_players": ["Large trading houses", "Freight forwarders"],
                    "startup_competition": "Low",
                    "international_threat": "High",
                    "differentiation_opportunities": ["Tech automation", "Niche product focus", "End-to-end solutions"]
                },
                growth_trajectory="growing",
                regulatory_requirements=[
                    "Import/Export license",
                    "Customs clearance authorization",
                    "Foreign exchange compliance",
                    "Product-specific certifications"
                ]
            ),
            
            "digital_services": SectorOpportunity(
                sector_name="Digital Services",
                market_conditions=MarketConditions(
                    demand_level=9.0,
                    competition_intensity=8.5,
                    market_maturity="growth",
                    entry_barriers=3.0,
                    regulatory_complexity=2.0,
                    capital_requirements={"min": 100000, "recommended": 500000, "optimal": 2000000},
                    customer_acquisition_difficulty=5.0
                ),
                required_skills={
                    "digital_marketing": ExperienceLevel.ADVANCED,
                    "web_development": ExperienceLevel.INTERMEDIATE,
                    "client_management": ExperienceLevel.INTERMEDIATE,
                    "project_management": ExperienceLevel.INTERMEDIATE,
                    "content_creation": ExperienceLevel.INTERMEDIATE
                },
                typical_business_models=[
                    {
                        "model": "Agency Services",
                        "revenue_potential": "LKR 1M-20M annually",
                        "time_to_revenue": "1-3 months",
                        "capital_intensity": "low"
                    },
                    {
                        "model": "SaaS Products",
                        "revenue_potential": "LKR 2M-100M annually",
                        "time_to_revenue": "6-18 months",
                        "capital_intensity": "moderate"
                    }
                ],
                success_factors=[
                    "Strong portfolio and case studies",
                    "Excellent client relationships",
                    "Staying current with technology trends",
                    "Efficient delivery processes",
                    "Strong personal brand"
                ],
                common_failure_reasons=[
                    "Underpricing services",
                    "Poor client communication",
                    "Inability to scale",
                    "Lack of specialization",
                    "Cash flow management issues"
                ],
                competitive_landscape={
                    "established_players": ["Large agencies", "Freelancers"],
                    "startup_competition": "Very High",
                    "international_threat": "High",
                    "differentiation_opportunities": ["Local market expertise", "Niche specialization", "Premium positioning"]
                },
                growth_trajectory="booming",
                regulatory_requirements=[
                    "Basic business registration",
                    "Tax compliance",
                    "Data protection (if applicable)"
                ]
            )
        }
    
    def analyze_fit(self, profile: EntrepreneurProfile, sector_name: str) -> Dict[str, Any]:
        """Enhanced fit analysis with more accurate scoring"""
        if sector_name not in self.sectors:
            return {"error": f"Sector {sector_name} not found"}
        
        sector = self.sectors[sector_name]
        readiness = profile.calculate_readiness_score()
        
        # Calculate detailed fit scores
        skill_fit = self._calculate_skill_fit(profile, sector)
        financial_fit = self._calculate_financial_fit(profile, sector)
        market_fit = self._calculate_market_fit(profile, sector)
        risk_fit = self._calculate_risk_fit(profile, sector)
        timeline_fit = self._calculate_timeline_fit(profile, sector)
        
        # Weighted overall fit calculation
        overall_fit = (
            skill_fit["score"] * 0.3 +
            financial_fit["score"] * 0.25 +
            market_fit["score"] * 0.2 +
            risk_fit["score"] * 0.15 +
            timeline_fit["score"] * 0.1
        )
        
        # Generate realistic projections
        projections = self._generate_realistic_projections(profile, sector, overall_fit)
        
        # Generate specific recommendations
        recommendations = self._generate_detailed_recommendations(profile, sector, overall_fit)
        
        return {
            "sector": sector_name,
            "overall_fit_score": round(overall_fit, 1),
            "sector_attractiveness": sector.calculate_sector_attractiveness(),
            "entrepreneur_readiness": readiness,
            "detailed_fit_analysis": {
                "skill_fit": skill_fit,
                "financial_fit": financial_fit,
                "market_fit": market_fit,
                "risk_fit": risk_fit,
                "timeline_fit": timeline_fit
            },
            "realistic_projections": projections,
            "success_probability": self._calculate_success_probability(overall_fit, readiness["overall"]),
            "recommendations": recommendations,
            "next_steps": self._generate_specific_next_steps(profile, sector, overall_fit),
            "risk_mitigation": self._generate_risk_mitigation_strategies(profile, sector)
        }
    
    def _calculate_skill_fit(self, profile: EntrepreneurProfile, sector: SectorOpportunity) -> Dict[str, Any]:
        """Calculate how well skills match sector requirements"""
        skill_gaps = []
        skill_matches = []
        total_score = 0

        required_skills = sector.required_skills

        # Map ExperienceLevel to numeric values
        level_map = {
            ExperienceLevel.BEGINNER: 1,
            ExperienceLevel.INTERMEDIATE: 2,
            ExperienceLevel.ADVANCED: 3,
            ExperienceLevel.EXPERT: 4
        }

        for skill, required_level in required_skills.items():
            # Check across all skill categories
            current_level = None
            for category in ["technical_skills", "business_skills", "industry_knowledge", "soft_skills"]:
                skill_dict = getattr(profile.skills, category, {})
                if skill in skill_dict:
                    current_level = skill_dict[skill]
                    break

            if current_level is None:
                skill_gaps.append(f"{skill} - Not present (Need: {required_level.value})")
                total_score += 0
            else:
                current_num = level_map.get(current_level, 0)
                required_num = level_map.get(required_level, 1)
                if current_num < required_num:
                    skill_gaps.append(f"{skill} - {current_level.value} (Need: {required_level.value})")
                    total_score += (current_num / required_num) * 2
                else:
                    skill_matches.append(f"{skill} - {current_level.value} (Exceeds requirement)")
                    total_score += 2

        max_score = len(required_skills) * 2
        skill_score = (total_score / max_score) * 10 if max_score > 0 else 0

        return {
            "score": round(skill_score, 1),
            "skill_matches": skill_matches,
            "skill_gaps": skill_gaps,
            "development_needed": len(skill_gaps) > len(skill_matches),
            "critical_gaps": [gap for gap in skill_gaps if "expert" in gap.lower() or "advanced" in gap.lower()]
        }
    
    def _calculate_financial_fit(self, profile: EntrepreneurProfile, sector: SectorOpportunity) -> Dict[str, Any]:
        """Calculate financial readiness for sector"""
        capital_req = sector.market_conditions.capital_requirements
        available_capital = profile.available_capital
        
        # Capital adequacy scoring
        if available_capital >= capital_req["optimal"]:
            capital_score = 10
            capital_status = "Excellent - Can pursue premium positioning"
        elif available_capital >= capital_req["recommended"]:
            capital_score = 8
            capital_status = "Good - Should enable solid market entry"
        elif available_capital >= capital_req["min"]:
            capital_score = 6
            capital_status = "Adequate - Will need careful resource management"
        elif available_capital >= capital_req["min"] * 0.7:
            capital_score = 4
            capital_status = "Below optimal - Consider partnerships or phased approach"
        else:
            capital_score = 2
            capital_status = "Insufficient - Need funding or different sector"
        
        # Monthly sustainability check
        monthly_burn = profile.monthly_personal_expenses + profile.family_financial_obligations
        months_runway = available_capital / monthly_burn if monthly_burn > 0 else float('inf')
        
        runway_score = min(months_runway / 12, 1) * 10  # 12 months ideal
        
        overall_financial_score = (capital_score * 0.7 + runway_score * 0.3)
        
        return {
            "score": round(overall_financial_score, 1),
            "capital_adequacy": capital_status,
            "months_runway": round(months_runway, 1),
            "recommended_funding": capital_req["recommended"] - available_capital if available_capital < capital_req["recommended"] else 0,
            "financial_warnings": self._generate_financial_warnings(profile, sector)
        }
    
    def _calculate_market_fit(self, profile: EntrepreneurProfile, sector: SectorOpportunity) -> Dict[str, Any]:
        """Calculate market positioning potential"""
        # Network strength for customer acquisition
        network_score = profile.existing_network_strength
        
        # Brand credibility
        brand_score = profile.personal_brand_strength
        
        # Local market advantages
        local_advantages = []
        if profile.local_market_understanding in [ExperienceLevel.ADVANCED, ExperienceLevel.EXPERT]:
            local_advantages.append("Strong local market knowledge")
        
        if len([l for l in profile.language_skills.values() if l in [ExperienceLevel.ADVANCED, ExperienceLevel.EXPERT]]) >= 2:
            local_advantages.append("Multilingual communication advantage")
        
        if profile.cultural_adaptability >= 7:
            local_advantages.append("High cultural adaptability")
        
        # Competition assessment
        competition_challenge = sector.market_conditions.competition_intensity
        market_score = (network_score * 0.3 + brand_score * 0.3 + 
                       (10 - competition_challenge) * 0.2 + len(local_advantages) * 0.2)
        
        return {
            "score": round(market_score, 1),
            "local_advantages": local_advantages,
            "network_strength": f"{network_score}/10",
            "brand_strength": f"{brand_score}/10",
            "competition_level": f"{competition_challenge}/10",
            "market_entry_strategy": self._suggest_market_entry_strategy(profile, sector)
        }
    
    def _calculate_risk_fit(self, profile: EntrepreneurProfile, sector: SectorOpportunity) -> Dict[str, Any]:
        """Calculate risk tolerance alignment"""
        sector_risk_level = (
            sector.market_conditions.competition_intensity * 0.3 +
            sector.market_conditions.regulatory_complexity * 0.25 +
            sector.market_conditions.entry_barriers * 0.2 +
            sector.market_conditions.customer_acquisition_difficulty * 0.25
        )
        
        risk_tolerance_scores = {"conservative": 3, "moderate": 6, "aggressive": 9}
        entrepreneur_risk_tolerance = risk_tolerance_scores.get(profile.risk_tolerance, 5)
        
        # Financial buffer assessment
        buffer_months = profile.emergency_fund_months
        buffer_score = min(buffer_months / 6, 1) * 3  # 6 months ideal
        
        risk_score = 10 - abs(entrepreneur_risk_tolerance - sector_risk_level) + buffer_score
        risk_score = max(0, min(risk_score, 10))
        
        return {
            "score": round(risk_score, 1),
            "sector_risk_level": f"{sector_risk_level:.1f}/10",
            "your_risk_tolerance": f"{entrepreneur_risk_tolerance}/10",
            "financial_buffer": f"{buffer_months} months",
            "risk_alignment": "Good" if abs(entrepreneur_risk_tolerance - sector_risk_level) <= 2 else "Misaligned",
            "risk_factors": sector.common_failure_reasons
        }
    
    def _calculate_timeline_fit(self, profile: EntrepreneurProfile, sector: SectorOpportunity) -> Dict[str, Any]:
        """Calculate timeline compatibility"""
        timeline_needs = {
            "immediate": 1,
            "3-6months": 6,
            "1year+": 12
        }
        
        entrepreneur_timeline = timeline_needs.get(profile.income_timeline_need, 6)
        
        # Get typical time to revenue for sector
        typical_times = []
        for model in sector.typical_business_models:
            time_str = model["time_to_revenue"]
            if "1-3 months" in time_str:
                typical_times.append(2)
            elif "6-12 months" in time_str:
                typical_times.append(9)
            elif "12-24 months" in time_str:
                typical_times.append(18)
            else:
                typical_times.append(12)  # default
        
        avg_sector_timeline = sum(typical_times) / len(typical_times) if typical_times else 12
        
        if entrepreneur_timeline >= avg_sector_timeline:
            timeline_score = 10
        elif entrepreneur_timeline >= avg_sector_timeline * 0.7:
            timeline_score = 7
        else:
            timeline_score = 4
        
        return {
            "score": timeline_score,
            "your_timeline_need": profile.income_timeline_need,
            "sector_typical_timeline": f"{avg_sector_timeline:.0f} months",
            "timeline_compatibility": "Good" if timeline_score >= 7 else "Challenging",
            "timeline_recommendations": self._generate_timeline_recommendations(profile, sector)
        }
    
    def _generate_realistic_projections(self, profile: EntrepreneurProfile, sector: SectorOpportunity, fit_score: float) -> Dict[str, Any]:
        """Generate realistic financial projections based on fit"""
        base_models = sector.typical_business_models
        
        # Adjust projections based on fit score
        fit_multiplier = fit_score / 10
        
        projections = []
        for model in base_models:
            # Parse revenue potential
            revenue_str = model["revenue_potential"]
            if "LKR" in revenue_str and "-" in revenue_str:
                # Extract min and max values
                parts = revenue_str.replace("LKR", "").replace("annually", "").strip()
                min_max = parts.split("-")
                if len(min_max) == 2:
                    min_val = self._parse_revenue_value(min_max[0])
                    max_val = self._parse_revenue_value(min_max[1])
                    
                    # Adjust based on fit
                    adjusted_min = min_val * max(0.3, fit_multiplier * 0.7)
                    adjusted_max = max_val * min(1.2, fit_multiplier * 1.1)
                    
                    projections.append({
                        "business_model": model["model"],
                        "year_1_revenue_range": f"LKR {adjusted_min:,.0f} - {adjusted_max * 0.4:,.0f}",
                        "year_2_revenue_range": f"LKR {adjusted_min * 1.5:,.0f} - {adjusted_max * 0.7:,.0f}",
                        "year_3_revenue_range": f"LKR {adjusted_min * 2:,.0f} - {adjusted_max:,.0f}",
                        "time_to_first_revenue": model["time_to_revenue"],
                        "probability_of_success": self._calculate_model_success_probability(profile, model, fit_score)
                    })
        
        return {
            "projections_by_model": projections,
            "assumptions": [
                "Assumes consistent market conditions",
                "Requires full-time commitment after initial validation",
                "Success depends on execution quality",
                f"Projections adjusted based on {fit_score:.1f}/10 fit score"
            ],
            "confidence_level": "Medium" if fit_score >= 6 else "Low"
        }
    
    def _parse_revenue_value(self, value_str: str) -> float:
        """Parse revenue values like '2M', '50K' etc"""
        value_str = value_str.strip().upper()
        if 'M' in value_str:
            return float(value_str.replace('M', '')) * 1000000
        elif 'K' in value_str:
            return float(value_str.replace('K', '')) * 1000
        else:
            return float(value_str)
    
    def _calculate_model_success_probability(self, profile: EntrepreneurProfile, model: Dict[str, Any], fit_score: float) -> str:
        """Calculate probability of success for specific business model"""
        base_probability = fit_score * 10  # Convert to percentage
        
        # Adjust based on model characteristics
        if model["capital_intensity"] == "low" and profile.available_capital >= 500000:
            base_probability += 10
        elif model["capital_intensity"] == "high" and profile.available_capital < 5000000:
            base_probability -= 20
        
        # Cap at realistic ranges
        probability = max(10, min(base_probability, 70))
        
        if probability >= 50:
            return f"{probability:.0f}% (High)"
        elif probability >= 30:
            return f"{probability:.0f}% (Medium)"
        else:
            return f"{probability:.0f}% (Low)"
    
    def _calculate_success_probability(self, fit_score: float, readiness_score: float) -> Dict[str, Any]:
        """Calculate overall probability of business success"""
        combined_score = (fit_score + readiness_score) / 2
        
        # Research-based success probability calculations
        if combined_score >= 8:
            success_prob = 60
            risk_level = "Low"
        elif combined_score >= 6.5:
            success_prob = 40  
            risk_level = "Medium"
        elif combined_score >= 5:
            success_prob = 25
            risk_level = "High"
        else:
            success_prob = 15
            risk_level = "Very High"
        
        return {
            "overall_success_probability": f"{success_prob}%",
            "risk_level": risk_level,
            "key_success_factors": [
                "Market validation before significant investment",
                "Strong execution on core business fundamentals",
                "Adequate capital management and runway",
                "Customer acquisition and retention",
                "Adaptation to market feedback"
            ],
            "probability_breakdown": {
                "survive_year_1": f"{success_prob + 20}%",
                "achieve_profitability": f"{success_prob}%", 
                "scale_significantly": f"{max(10, success_prob - 20)}%"
            }
        }
    
    def _generate_detailed_recommendations(self, profile: EntrepreneurProfile, sector: SectorOpportunity, fit_score: float) -> List[Dict[str, Any]]:
        """Generate specific, actionable recommendations"""
        recommendations = []
        
        # Fit-based recommendations
        if fit_score >= 7.5:
            recommendations.append({
                "priority": "HIGH",
                "category": "Action",
                "recommendation": "Proceed with market validation immediately",
                "rationale": "Strong fit indicates high potential for success",
                "timeline": "Start this week"
            })
        elif fit_score >= 5.5:
            recommendations.append({
                "priority": "MEDIUM", 
                "category": "Development",
                "recommendation": "Address skill gaps before full commitment",
                "rationale": "Good foundation but needs strengthening",
                "timeline": "2-6 months preparation"
            })
        else:
            recommendations.append({
                "priority": "LOW",
                "category": "Alternative",
                "recommendation": "Consider different sector or significant preparation",
                "rationale": "Current fit is below success threshold",
                "timeline": "6+ months development needed"
            })
        
        # Financial recommendations
        if profile.available_capital < sector.market_conditions.capital_requirements["min"]:
            recommendations.append({
                "priority": "CRITICAL",
                "category": "Funding",
                "recommendation": "Secure additional funding before starting",
                "rationale": "Insufficient capital for sustainable launch",
                "timeline": "Before any business activities"
            })
        
        # Skill-specific recommendations
        required_skills = sector.required_skills
        for skill, required_level in required_skills.items():
            current_level = self._get_current_skill_level(profile, skill)
            if current_level is None or current_level.value < required_level.value:
                recommendations.append({
                    "priority": "HIGH" if required_level in [ExperienceLevel.ADVANCED, ExperienceLevel.EXPERT] else "MEDIUM",
                    "category": "Skill Development",
                    "recommendation": f"Develop {skill} to {required_level.value} level",
                    "rationale": f"Critical skill gap identified",
                    "timeline": "1-6 months depending on current level"
                })
        
        return recommendations
    
    def _get_current_skill_level(self, profile: EntrepreneurProfile, skill: str) -> Optional[ExperienceLevel]:
        """Get current level of specific skill across all categories"""
        for category in ["technical_skills", "business_skills", "industry_knowledge", "soft_skills"]:
            skill_dict = getattr(profile.skills, category, {})
            if skill in skill_dict:
                return skill_dict[skill]
        return None
    
    def _generate_specific_next_steps(self, profile: EntrepreneurProfile, sector: SectorOpportunity, fit_score: float) -> List[Dict[str, Any]]:
        """Generate specific next steps with timelines and costs"""
        next_steps = []
        
        if fit_score >= 6:
            # Validation phase steps
            next_steps.extend([
                {
                    "step": "Market Research & Customer Interviews",
                    "description": "Interview 20-30 potential customers about their pain points",
                    "timeline": "Weeks 1-3",
                    "estimated_cost": "LKR 15,000 - 30,000",
                    "success_criteria": "Identify consistent pain points in 60%+ of interviews"
                },
                {
                    "step": "Competitive Analysis",
                    "description": "Analyze top 10 competitors in detail",
                    "timeline": "Weeks 2-4", 
                    "estimated_cost": "LKR 5,000 - 10,000",
                    "success_criteria": "Identify clear differentiation opportunities"
                },
                {
                    "step": "MVP Development",
                    "description": "Build minimum viable product or service prototype",
                    "timeline": "Weeks 4-8",
                    "estimated_cost": "LKR 50,000 - 200,000",
                    "success_criteria": "Get positive feedback from 10+ test users"
                }
            ])
        else:
            # Preparation phase steps
            next_steps.extend([
                {
                    "step": "Skill Gap Analysis & Development Plan",
                    "description": "Identify and address critical skill gaps",
                    "timeline": "Weeks 1-2",
                    "estimated_cost": "LKR 2,000 - 5,000",
                    "success_criteria": "Clear development plan with timelines"
                },
                {
                    "step": "Industry Immersion",
                    "description": "Deep dive into sector through courses, networking, shadowing",
                    "timeline": "Weeks 3-12",
                    "estimated_cost": "LKR 25,000 - 100,000",
                    "success_criteria": "Build industry knowledge and network"
                }
            ])
        
        # Universal steps
        next_steps.extend([
            {
                "step": "Legal & Regulatory Research",
                "description": "Understand all compliance requirements",
                "timeline": "Weeks 1-2",
                "estimated_cost": "LKR 10,000 - 25,000",
                "success_criteria": "Complete regulatory checklist"
            },
            {
                "step": "Financial Planning",
                "description": "Detailed business plan with financial projections",
                "timeline": "Weeks 3-5",
                "estimated_cost": "LKR 5,000 - 15,000",
                "success_criteria": "Realistic 3-year financial model"
            }
        ])
        
        return next_steps
    
    def _generate_risk_mitigation_strategies(self, profile: EntrepreneurProfile, sector: SectorOpportunity) -> List[Dict[str, Any]]:
        """Generate specific risk mitigation strategies"""
        strategies = []
        
        # Financial risk mitigation
        if profile.emergency_fund_months < 6:
            strategies.append({
                "risk": "Financial runway depletion",
                "strategy": "Maintain part-time income during early stages",
                "implementation": "Freelance or consult in current field while building business",
                "cost": "Opportunity cost of slower growth"
            })
        
        # Market risk mitigation
        if sector.market_conditions.competition_intensity >= 7:
            strategies.append({
                "risk": "High competition pressure",
                "strategy": "Focus on underserved niche markets initially",
                "implementation": "Identify specific customer segments competitors ignore",
                "cost": "Smaller initial market size"
            })
        
        # Skill risk mitigation
        critical_skills = [skill for skill, level in sector.required_skills.items() 
                         if level in [ExperienceLevel.ADVANCED, ExperienceLevel.EXPERT]]
        if critical_skills:
            strategies.append({
                "risk": "Critical skill gaps",
                "strategy": "Partner with or hire experts in missing areas",
                "implementation": "Form strategic partnerships or early key hires",
                "cost": "Equity dilution or higher salary costs"
            })
        
        # Regulatory risk mitigation
        if sector.market_conditions.regulatory_complexity >= 7:
            strategies.append({
                "risk": "Regulatory compliance failures",
                "strategy": "Engage regulatory consultant from day one",
                "implementation": "Budget 5-10% of capital for compliance support",
                "cost": "LKR 100,000 - 500,000 annually"
            })
        
        return strategies
    
    def _generate_financial_warnings(self, profile: EntrepreneurProfile, sector: SectorOpportunity) -> List[str]:
        """Generate financial warnings and alerts"""
        warnings = []
        
        monthly_burn = profile.monthly_personal_expenses + profile.family_financial_obligations
        runway_months = profile.available_capital / monthly_burn if monthly_burn > 0 else float('inf')
        
        if runway_months < 12:
            warnings.append(f"Only {runway_months:.1f} months personal runway - consider keeping current income initially")
        
        if profile.debt_obligations > profile.existing_income * 0.3:
            warnings.append("High debt burden may pressure quick revenue generation - could lead to poor decisions")
        
        if profile.available_capital < sector.market_conditions.capital_requirements["min"]:
            shortfall = sector.market_conditions.capital_requirements["min"] - profile.available_capital
            warnings.append(f"Need additional LKR {shortfall:,.0f} for minimum viable launch")
        
        if profile.emergency_fund_months < 3:
            warnings.append("Insufficient emergency fund - extremely risky to start business without safety net")
        
        return warnings
    
    def _suggest_market_entry_strategy(self, profile: EntrepreneurProfile, sector: SectorOpportunity) -> str:
        """Suggest optimal market entry strategy"""
        if profile.existing_network_strength >= 7:
            return "Network-leveraged entry: Use existing relationships for initial customers and credibility"
        elif profile.available_capital >= sector.market_conditions.capital_requirements["recommended"]:
            return "Capital-intensive entry: Invest in strong market position from launch"
        elif sector.market_conditions.competition_intensity <= 5:
            return "Direct entry: Market has space for new players with good execution"
        else:
            return "Niche entry: Focus on underserved segment before expanding to broader market"
    
    def _generate_timeline_recommendations(self, profile: EntrepreneurProfile, sector: SectorOpportunity) -> List[str]:
        """Generate timeline-specific recommendations"""
        recommendations = []
        
        if profile.income_timeline_need == "immediate":
            recommendations.extend([
                "Consider consulting/freelancing in your expertise area for immediate income",
                "Look for business models with fastest time-to-revenue",
                "Avoid capital-intensive models that require long development cycles"
            ])
        elif profile.income_timeline_need == "3-6months":
            recommendations.extend([
                "Focus on service-based offerings initially",
                "Validate with manual processes before automating",
                "Build customer relationships that can expand over time"
            ])
        else:  # 1year+
            recommendations.extend([
                "Can pursue more complex, higher-value opportunities",
                "Take time for proper market research and product development",
                "Consider partnerships with established players for faster market entry"
            ])
        
        return recommendations

class Qwen3RAGPipeline:
    """
    Placeholder for Qwen3 8B + RAG pipeline.
    In production, connect to your Qwen3 model and retrieval backend (e.g., FAISS, ChromaDB, or API).
    """
    def __init__(self, knowledge_base_path: str = "knowledge_base/"):
        self.knowledge_base_path = knowledge_base_path
        self.documents = self._load_documents()

    def _load_documents(self):
        docs = []
        if not os.path.exists(self.knowledge_base_path):
            return docs
        for fname in os.listdir(self.knowledge_base_path):
            if fname.endswith(".txt"):
                with open(os.path.join(self.knowledge_base_path, fname), "r", encoding="utf-8") as f:
                    docs.append({"title": fname, "content": f.read()})
        return docs

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        # Simple keyword search for demonstration; replace with vector search for production
        results = []
        for doc in self.documents:
            if query.lower() in doc["content"].lower():
                results.append(doc["content"][:500])
        return results[:top_k]

    def generate(self, prompt: str, context: List[str]) -> str:
        # In production, call Qwen3 8B LLM with context and prompt
        # Here, we simulate by concatenating context and prompt
        context_str = "\n\n".join(context)
        # Replace this with actual Qwen3 LLM call
        return f"Context:\n{context_str}\n\nPrompt:\n{prompt}\n\n[Qwen3 8B LLM output here]"

        


class EnhancedBusinessDiscoveryPlatform:
    """Main platform with enhanced analysis capabilities and Qwen3+RAG support"""

    def __init__(self):
        self.analyzer = EnhancedOpportunityAnalyzer()
        self.profile: Optional[EntrepreneurProfile] = None
        self.rag_pipeline = Qwen3RAGPipeline(knowledge_base_path="knowledge_base/")

    def set_profile(self, profile: EntrepreneurProfile):
        """Set entrepreneur profile"""
        self.profile = profile
        logger.info(f"Profile set for {profile.name}")

    def discover_opportunities(self) -> Dict[str, Any]:
        """Comprehensive opportunity discovery with enhanced analysis and RAG"""
        if not self.profile:
            return {"error": "No profile set"}

        # Use RAG to enrich the opportunity discovery
        query = f"Best business opportunities for a {self.profile.age_range} entrepreneur in {self.profile.location} with skills: {list(self.profile.skills.technical_skills.keys()) + list(self.profile.skills.business_skills.keys())}"
        retrieved_contexts = self.rag_pipeline.retrieve(query, top_k=3)
        prompt = (
            f"Given the entrepreneur profile: {self.profile.name}, skills: {self.profile.skills}, "
            f"location: {self.profile.location}, and Sri Lankan market context, "
            f"analyze and recommend the most suitable business sectors and paths. "
            f"Provide sector fit, financial projections, and actionable steps."
        )
        # This would be the LLM output in a real pipeline
        llm_output = self.rag_pipeline.generate(prompt, retrieved_contexts)

        # Continue with your structured analysis
        readiness = self.profile.calculate_readiness_score()
        sector_analyses = {}
        for sector_name in self.analyzer.sectors.keys():
            analysis = self.analyzer.analyze_fit(self.profile, sector_name)
            sector_analyses[sector_name] = analysis
        ranked_opportunities = self._rank_opportunities(sector_analyses)

        results = {
            "entrepreneur": self.profile.name,
            "location": self.profile.location,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "entrepreneur_readiness": readiness,
            "sector_analyses": sector_analyses,
            "ranked_opportunities": ranked_opportunities,
            "overall_recommendations": self._generate_overall_recommendations(readiness, ranked_opportunities),
            "immediate_action_plan": self._generate_action_plan(ranked_opportunities),
            "funding_strategy": self._generate_funding_strategy(),
            "risk_assessment": self._generate_overall_risk_assessment(sector_analyses),
            "success_timeline": self._generate_success_timeline(ranked_opportunities),
            "rag_llm_output": llm_output
        }
        return results
    
    def _generate_action_plan(self, ranked_opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate immediate action plan based on top opportunities"""
        if not ranked_opportunities:
            return []
        
        top_opportunity = ranked_opportunities[0]
        actions = []
        
        if top_opportunity["combined_score"] >= 7:
            actions.extend([
                {
                    "priority": "HIGH",
                    "action": "Start market validation for " + top_opportunity["sector"].replace('_', ' ').title(),
                    "timeline": "This week",
                    "outcome": "Validate customer demand and pain points"
                },
                {
                    "priority": "MEDIUM", 
                    "action": "Complete competitive analysis",
                    "timeline": "Within 2 weeks",
                    "outcome": "Understand market positioning opportunities"
                }
            ])
        elif top_opportunity["combined_score"] >= 5:
            actions.extend([
                {
                    "priority": "HIGH",
                    "action": "Address critical skill gaps",
                    "timeline": "1-3 months",
                    "outcome": "Improve sector readiness score"
                },
                {
                    "priority": "MEDIUM",
                    "action": "Conduct deep sector research",
                    "timeline": "2-4 weeks", 
                    "outcome": "Better understanding of requirements and opportunities"
                }
            ])
        else:
            actions.append({
                "priority": "LOW",
                "action": "Focus on general entrepreneurial skill development",
                "timeline": "3-6 months",
                "outcome": "Improve overall readiness before sector selection"
            })
        
        return actions

    def _generate_funding_strategy(self) -> Dict[str, Any]:
        """Generate funding strategy recommendations"""
        if not self.profile:
            return {}
        
        available_capital = self.profile.available_capital
        strategies = []
        
        if available_capital < 500000:
            strategies.append({
                "strategy": "Bootstrap + Part-time Income",
                "amount": "LKR 0-200K additional",
                "pros": ["Full control", "No debt", "Learn lean operations"],
                "cons": ["Slower growth", "Limited resources", "High personal risk"],
                "timeline": "Immediate"
            })
        
        if available_capital < 2000000:
            strategies.append({
                "strategy": "Friends & Family Round",
                "amount": "LKR 500K-2M",
                "pros": ["Easier to secure", "Flexible terms", "Supportive investors"],
                "cons": ["Limited amount", "Personal relationships at risk"],
                "timeline": "2-4 months"
            })
        
        if available_capital >= 1000000:
            strategies.append({
                "strategy": "Angel Investment",
                "amount": "LKR 2M-10M", 
                "pros": ["Significant capital", "Mentor support", "Network access"],
                "cons": ["Equity dilution", "Investor pressure", "Due diligence required"],
                "timeline": "3-6 months"
            })
        
        return {
            "recommended_strategies": strategies,
            "preparation_required": [
                "Solid business plan with financial projections",
                "Market validation evidence", 
                "Clear use of funds breakdown",
                "Legal structure setup"
            ]
        }

    def _generate_overall_risk_assessment(self, sector_analyses: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate comprehensive risk assessment across all sectors"""
        financial_risks = []
        market_risks = []
        operational_risks = []
        personal_risks = []
        
        if not self.profile:
            return {}
        
        # Financial risks
        if self.profile.emergency_fund_months < 6:
            financial_risks.append("Insufficient emergency fund for business risk")
        if self.profile.debt_obligations > self.profile.existing_income * 0.3:
            financial_risks.append("High debt burden may pressure quick returns")
        
        # Market risks
        high_competition_sectors = [s for s, a in sector_analyses.items() 
                                  if a["sector_attractiveness"] < 6]
        if high_competition_sectors:
            market_risks.append(f"High competition in preferred sectors: {', '.join(high_competition_sectors)}")
        
        # Operational risks
        if self.profile.time_available_per_week < 30:
            operational_risks.append("Limited time availability may slow progress")
        
        # Personal risks
        if self.profile.family_commitments in ["moderate", "heavy"]:
            personal_risks.append("Family commitments may limit business focus")
        
        return {
            "financial_risks": financial_risks,
            "market_risks": market_risks, 
            "operational_risks": operational_risks,
            "personal_risks": personal_risks
        }

    def _generate_success_timeline(self, ranked_opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate realistic success timeline and milestones"""
        if not ranked_opportunities:
            return {}

        top_opportunity = ranked_opportunities[0]

        timeline = {
            "month_1_3": "Market validation, competitive analysis, MVP development",
            "month_4_6": "Customer acquisition, product iteration, revenue generation",
            "month_7_12": "Scale operations, hire team, establish market position",
            "year_2": "Market expansion, product diversification, sustainable growth",
            "year_3": "Market leadership, strategic partnerships, exit planning"
        }

        critical_milestones = [
            "First paying customer within 3 months",
            "Break-even within 8-12 months",
            "LKR 1M+ annual revenue by month 18",
            "Sustainable team of 3+ people by year 2"
        ]

        success_metrics = [
            "Monthly recurring revenue growth",
            "Customer acquisition cost vs lifetime value",
            "Market share in target segment",
            "Team productivity and retention",
            "Founder salary replacement"
        ]

        return {
            "timeline": timeline,
            "critical_milestones": critical_milestones,
            "success_metrics": success_metrics
        }
        
        # Use RAG to enrich the opportunity discovery


    def _rank_opportunities(self, sector_analyses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank business opportunities based on combined fit and sector attractiveness.
        Returns a list of dicts with sector, scores, and key points.
        """
        ranked = []
        for sector, analysis in sector_analyses.items():
            combined_score = round(
                (analysis["overall_fit_score"] * 0.6 + analysis["sector_attractiveness"] * 0.4), 1
            )
            ranked.append({
                "sector": sector,
                "fit_score": analysis["overall_fit_score"],
                "sector_attractiveness": analysis["sector_attractiveness"],
                "combined_score": combined_score,
                "success_probability": analysis["success_probability"]["overall_success_probability"],
                "recommendation_level": (
                    "Strongly Recommended" if combined_score >= 7.5 else
                    "Recommended" if combined_score >= 6 else
                    "Consider with Caution"
                ),
                "key_advantages": analysis["detailed_fit_analysis"]["market_fit"]["local_advantages"],
                "main_concerns": analysis["detailed_fit_analysis"]["skill_fit"]["critical_gaps"],
            })
        # Sort by combined score descending
        ranked.sort(key=lambda x: x["combined_score"], reverse=True)
        return ranked

    def _generate_overall_recommendations(self, readiness: Dict[str, float], ranked_opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate overall strategic recommendations based on readiness and top opportunities.
        """
        recommendations = []
        top = ranked_opportunities[0] if ranked_opportunities else None

        if not top:
            recommendations.append({
                "type": "General",
                "recommendation": "No strong business path identified.",
                "rationale": "No sector scored high enough for recommendation.",
                "action": "Consider skill development or alternative sectors."
            })
            return recommendations

        if top["combined_score"] >= 7.5:
            recommendations.append({
                "type": "Primary",
                "recommendation": f"Strongly pursue {top['sector'].replace('_', ' ').title()} sector.",
                "rationale": "Top sector has excellent fit and market attractiveness.",
                "action": "Begin validation and early market entry steps."
            })
        elif top["combined_score"] >= 6:
            recommendations.append({
                "type": "Primary",
                "recommendation": f"Consider {top['sector'].replace('_', ' ').title()} sector with preparation.",
                "rationale": "Good fit, but address key gaps before launch.",
                "action": "Focus on skill/financial gap closure, then proceed."
            })
        else:
            recommendations.append({
                "type": "Caution",
                "recommendation": "Do not commit to any sector yet.",
                "rationale": "No sector meets minimum fit threshold.",
                "action": "Prioritize readiness improvement."
            })

        # Add a readiness-based note
        if readiness["overall"] < 6:
            recommendations.append({
                "type": "Readiness",
                "recommendation": "Improve overall entrepreneurial readiness.",
                "rationale": "Low readiness may increase risk of failure.",
                "action": "Focus on financial stability, skill-building, and market research."
            })

        return recommendations

# ...rest of your code remains unchanged...

async def enhanced_main():
    """Enhanced main function with comprehensive analysis for accurate business path selection"""
    platform = EnhancedBusinessDiscoveryPlatform()
    
    # Create realistic profile
    profile = create_realistic_profile()
    platform.set_profile(profile)

    # Run opportunity discovery
    results = platform.discover_opportunities()

    # Prepare output for report
    output_lines = []
    output_lines.append("=" * 90)
    output_lines.append("🇱🇰 ENHANCED SRI LANKAN BUSINESS PATH DISCOVERY REPORT")
    output_lines.append("=" * 90)
    output_lines.append(f"Entrepreneur: {results['entrepreneur']} | Location: {results['location']}")
    output_lines.append(f"Analysis Date: {results['analysis_date']}")
    output_lines.append("")

    # Add Qwen3+RAG LLM output at the top for context
    output_lines.append("Qwen3 8B + RAG Pipeline Output:")
    output_lines.append(results.get("rag_llm_output", "No LLM output available."))
    output_lines.append("")

    # Readiness summary
    readiness = results["entrepreneur_readiness"]
    output_lines.append("ENTREPRENEUR READINESS SCORES (0-10):")
    for k, v in readiness.items():
        output_lines.append(f"  {k.title()}: {v}")
    output_lines.append("")

    # Ranked opportunities
    output_lines.append("RANKED BUSINESS OPPORTUNITIES:")
    for i, opp in enumerate(results["ranked_opportunities"], 1):
        output_lines.append(f"{i}. {opp['sector'].replace('_', ' ').title()} | Combined Score: {opp['combined_score']}/10")
        output_lines.append(f"   Fit Score: {opp['fit_score']}/10 | Attractiveness: {opp['sector_attractiveness']}/10 | Success Probability: {opp['success_probability']}")
        output_lines.append(f"   Recommendation: {opp['recommendation_level']}")
        if opp['key_advantages']:
            output_lines.append(f"   Key Advantages: {', '.join(opp['key_advantages'])}")
        if opp['main_concerns']:
            output_lines.append(f"   Main Concerns: {', '.join(opp['main_concerns'])}")
        output_lines.append("")

    # Detailed sector analysis for top 2
    output_lines.append("=" * 60)
    output_lines.append("DETAILED SECTOR ANALYSIS (TOP 2)")
    output_lines.append("=" * 60)
    for opp in results["ranked_opportunities"][:2]:
        sector = opp['sector']
        analysis = results["sector_analyses"][sector]
        output_lines.append(f"\n--- {sector.replace('_', ' ').title()} ---")
        output_lines.append(f"Overall Fit Score: {analysis['overall_fit_score']}/10")
        output_lines.append(f"Sector Attractiveness: {analysis['sector_attractiveness']}/10")
        output_lines.append(f"Success Probability: {analysis['success_probability']['overall_success_probability']} (Risk: {analysis['success_probability']['risk_level']})")
        output_lines.append("Skill Fit:")
        output_lines.append(f"  Matches: {', '.join(analysis['detailed_fit_analysis']['skill_fit']['skill_matches']) or 'None'}")
        output_lines.append(f"  Gaps: {', '.join(analysis['detailed_fit_analysis']['skill_fit']['skill_gaps']) or 'None'}")
        output_lines.append("Financial Fit:")
        output_lines.append(f"  Capital Adequacy: {analysis['detailed_fit_analysis']['financial_fit']['capital_adequacy']}")
        output_lines.append(f"  Months Runway: {analysis['detailed_fit_analysis']['financial_fit']['months_runway']}")
        if analysis['detailed_fit_analysis']['financial_fit']['financial_warnings']:
            output_lines.append(f"  Warnings: {', '.join(analysis['detailed_fit_analysis']['financial_fit']['financial_warnings'])}")
        output_lines.append("Market Fit:")
        output_lines.append(f"  Local Advantages: {', '.join(analysis['detailed_fit_analysis']['market_fit']['local_advantages']) or 'None'}")
        output_lines.append(f"  Market Entry Strategy: {analysis['detailed_fit_analysis']['market_fit']['market_entry_strategy']}")
        output_lines.append("Risk Fit:")
        output_lines.append(f"  Alignment: {analysis['detailed_fit_analysis']['risk_fit']['risk_alignment']}")
        output_lines.append(f"  Risk Factors: {', '.join(analysis['detailed_fit_analysis']['risk_fit']['risk_factors'])}")
        output_lines.append("Timeline Fit:")
        output_lines.append(f"  Compatibility: {analysis['detailed_fit_analysis']['timeline_fit']['timeline_compatibility']}")
        output_lines.append(f"  Recommendations: {', '.join(analysis['detailed_fit_analysis']['timeline_fit']['timeline_recommendations'])}")
        output_lines.append("Projections:")
        for proj in analysis['realistic_projections']['projections_by_model']:
            output_lines.append(f"  Model: {proj['business_model']}")
            output_lines.append(f"    Year 1: {proj['year_1_revenue_range']}, Year 2: {proj['year_2_revenue_range']}, Year 3: {proj['year_3_revenue_range']}")
            output_lines.append(f"    Time to Revenue: {proj['time_to_first_revenue']}, Success Probability: {proj['probability_of_success']}")
        output_lines.append("Recommendations:")
        for rec in analysis['recommendations']:
            output_lines.append(f"  [{rec['priority']}] {rec['category']}: {rec['recommendation']} ({rec['timeline']})")
        output_lines.append("Next Steps:")
        for step in analysis['next_steps']:
            output_lines.append(f"  {step['step']}: {step['description']} ({step['timeline']}, Cost: {step['estimated_cost']})")
        output_lines.append("Risk Mitigation Strategies:")
        for strat in analysis['risk_mitigation']:
            output_lines.append(f"  {strat['risk']}: {strat['strategy']} (How: {strat['implementation']}, Cost: {strat['cost']})")

    # Overall recommendations
    output_lines.append("\n" + "=" * 60)
    output_lines.append("OVERALL STRATEGIC RECOMMENDATIONS")
    output_lines.append("=" * 60)
    for rec in results["overall_recommendations"]:
        output_lines.append(f"[{rec['type']}] {rec['recommendation']} - {rec['rationale']} (Action: {rec['action']})")

    # Immediate action plan
    output_lines.append("\nIMMEDIATE ACTION PLAN:")
    for act in results["immediate_action_plan"]:
        output_lines.append(f"  [{act['priority']}] {act['action']} ({act['timeline']}) -> {act['outcome']}")

    # Funding strategy
    output_lines.append("\nFUNDING STRATEGY:")
    funding = results["funding_strategy"]
    for strat in funding.get("recommended_strategies", []):
        output_lines.append(f"  {strat['strategy']}: {strat['amount']} (Pros: {', '.join(strat['pros'])}; Cons: {', '.join(strat['cons'])}; Timeline: {strat['timeline']})")
    output_lines.append("  Preparation Required: " + "; ".join(funding.get("preparation_required", [])))

    # Risk assessment
    output_lines.append("\nRISK ASSESSMENT:")
    risk = results["risk_assessment"]
    for k, v in risk.items():
        if v:
            output_lines.append(f"  {k.replace('_', ' ').title()}: {', '.join(v)}")

    # Success timeline
    output_lines.append("\nSUCCESS TIMELINE & MILESTONES:")
    timeline = results["success_timeline"]
    if timeline:
        for k, v in timeline["timeline"].items():
            output_lines.append(f"  {k.replace('_', ' ').title()}: {v}")
        output_lines.append("  Critical Milestones: " + "; ".join(timeline["critical_milestones"]))
        output_lines.append("  Success Metrics: " + "; ".join(timeline["success_metrics"]))

    # Save to text file
    with open("enhanced_business_path_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("Analysis complete. Results saved to enhanced_business_path_report.txt")

# Create a more realistic example profile
def create_realistic_profile() -> EntrepreneurProfile:
    """Create a realistic entrepreneur profile for demonstration"""
    
    skills = SkillAssessment(
        technical_skills={
            "programming": ExperienceLevel.INTERMEDIATE,
            "web_development": ExperienceLevel.INTERMEDIATE,
            "database_design": ExperienceLevel.BEGINNER,
            "cybersecurity": ExperienceLevel.BEGINNER
        },
        business_skills={
            "financial_analysis": ExperienceLevel.INTERMEDIATE,
            "marketing": ExperienceLevel.BEGINNER,
            "sales": ExperienceLevel.BEGINNER,
            "project_management": ExperienceLevel.INTERMEDIATE
        },
        industry_knowledge={
            "fintech": ExperienceLevel.BEGINNER,
            "e_commerce": ExperienceLevel.INTERMEDIATE,
            "digital_marketing": ExperienceLevel.INTERMEDIATE
        },
        soft_skills={
            "communication": ExperienceLevel.INTERMEDIATE,
            "leadership": ExperienceLevel.BEGINNER,
            "problem_solving": ExperienceLevel.ADVANCED,
            "adaptability": ExperienceLevel.INTERMEDIATE
        }
    )
    
    language_skills = {
        "english": ExperienceLevel.ADVANCED,
        "sinhala": ExperienceLevel.EXPERT,
        "tamil": ExperienceLevel.BEGINNER
    }
    
    return EntrepreneurProfile(
        name="Alex Perera",
        location="Colombo",
        age_range="25-30", 
        education_background="Computer Science degree",
        work_experience=["Software Developer - 2 years", "Freelance Web Development - 1 year"],
        skills=skills,
        
        # Financial situation - realistic for young professional
        available_capital=1500000.0,  # LKR 1.5M saved
        monthly_personal_expenses=60000.0,  # LKR 60K monthly expenses
        existing_income=120000.0,  # LKR 120K current salary
        family_financial_obligations=15000.0,  # LKR 15K family support
        debt_obligations=25000.0,  # LKR 25K education loan
        emergency_fund_months=4,  # 4 months of expenses saved
        
        # Risk and motivation
        risk_tolerance="moderate",
        motivation_level=8.0,
        time_available_per_week=35,  # Some evenings and weekends
        minimum_income_needed=80000.0,  # Need at least LKR 80K/month
        income_timeline_need="6-12months",
        
        # Personal constraints
        health_limitations=[],
        family_commitments="light",
        travel_willingness="national",
        relocation_willingness=False,
        partnership_preference="small-team",
        
        # Market position
        existing_network_strength=5.0,
        personal_brand_strength=4.0,
        credibility_factors=["CS degree", "Work experience", "Some freelance projects"],
        
        # Sri Lankan context
        language_skills=language_skills,
        cultural_adaptability=8.0,
        regulatory_knowledge=ExperienceLevel.BEGINNER,
        local_market_understanding=ExperienceLevel.INTERMEDIATE
    )

# To run the enhanced main function
if __name__ == "__main__":
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(enhanced_main())