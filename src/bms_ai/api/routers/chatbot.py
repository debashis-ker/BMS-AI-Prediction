from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, List, Any
import pandas as pd
from datetime import datetime, timedelta
from src.bms_ai.logger_config import setup_logger
import warnings
import os
from openai import OpenAI
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

log = setup_logger(__name__)
warnings.filterwarnings('ignore')

router = APIRouter(prefix="/chatbot", tags=["Energy Chatbot"])

# Initialize OpenAI client
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        log.warning("OPENAI_API_KEY not found in environment variables. ChatGPT features will be limited.")
        client = None
    else:
        client = OpenAI(api_key=openai_api_key)
        log.info("OpenAI client initialized successfully")
except Exception as e:
    log.error(f"Failed to initialize OpenAI client: {e}")
    client = None


class ChatRequest(BaseModel):
    question: str = Field(..., description="User's question about energy consumption")
    data: Dict[str, Any] = Field(..., description="BMS data for analysis")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context like time period, area filters")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Natural language answer")
    data: Optional[Dict[str, Any]] = Field(default={}, description="Supporting data/charts")
    recommendations: Optional[List[str]] = Field(default=[], description="Actionable recommendations")
    confidence: float = Field(..., description="Confidence score of the answer (0-1)")


class EnergyAnalyzer:
    """Core analysis engine for energy-related queries"""
    
    def __init__(self, data: Dict[str, Any]):
        self.raw_data = data
        self.df = self._prepare_dataframe()
    
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Convert API data to analyzed DataFrame"""
        try:
            if "queryResponse" not in self.raw_data:
                raise ValueError("Missing 'queryResponse' in data")
            
            records = self.raw_data["queryResponse"]
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            # Convert datetime
            if 'data_received_on' in df.columns:
                df['data_received_on'] = pd.to_datetime(df['data_received_on'])
                df['data_received_on'] = df['data_received_on'].dt.tz_localize(None)
            
            # Handle categorical mappings
            if 'monitoring_data' in df.columns:
                df['monitoring_data'] = df['monitoring_data'].replace({
                    'inactive': 0.0, 
                    'active': 1.0
                }).astype('float64', errors='ignore')
            
            return df
            
        except Exception as e:
            log.error(f"Error preparing dataframe: {e}")
            return pd.DataFrame()
    
    def get_energy_by_area(self) -> Dict[str, Any]:
        """Analyze which area consumes most electricity"""
        try:
            if self.df.empty:
                return {"error": "No data available"}
            
            # Filter for power consumption data
            power_df = self.df[
                (self.df['datapoint'] == 'Fan Power meter (KW)') |
                (self.df['datapoint'].str.contains('Power', case=False, na=False))
            ].copy()
            
            if power_df.empty:
                return {"error": "No power consumption data found"}
            
            # Convert monitoring_data to numeric
            power_df['monitoring_data'] = pd.to_numeric(power_df['monitoring_data'], errors='coerce')
            
            # Group by area/site
            if 'site' in power_df.columns:
                area_consumption = power_df.groupby('site')['monitoring_data'].agg([
                    ('total_consumption', 'sum'),
                    ('avg_consumption', 'mean'),
                    ('max_consumption', 'max'),
                    ('count', 'count')
                ]).round(2)
            elif 'equipment_name' in power_df.columns:
                area_consumption = power_df.groupby('equipment_name')['monitoring_data'].agg([
                    ('total_consumption', 'sum'),
                    ('avg_consumption', 'mean'),
                    ('max_consumption', 'max'),
                    ('count', 'count')
                ]).round(2)
            else:
                return {"error": "No area/site information available"}
            
            area_consumption = area_consumption.sort_values('total_consumption', ascending=False)
            
            # Get top consumer
            top_area = area_consumption.index[0]
            top_consumption = area_consumption.iloc[0]['total_consumption']
            avg_consumption = area_consumption.iloc[0]['avg_consumption']
            
            # Calculate percentage distribution
            total = area_consumption['total_consumption'].sum()
            area_consumption['percentage'] = (area_consumption['total_consumption'] / total * 100).round(2)
            
            return {
                "top_area": top_area,
                "top_consumption_kw": float(top_consumption),
                "avg_consumption_kw": float(avg_consumption),
                "percentage_of_total": float(area_consumption.iloc[0]['percentage']),
                "all_areas": area_consumption.to_dict('index'),
                "total_consumption": float(total)
            }
            
        except Exception as e:
            log.error(f"Error in get_energy_by_area: {e}")
            return {"error": str(e)}
    
    def get_reduction_recommendations(self, target_reduction_percent: float = 5.0) -> Dict[str, Any]:
        """Generate recommendations to reduce energy by target percentage"""
        try:
            if self.df.empty:
                return {"error": "No data available"}
            
            # Get current consumption baseline
            power_df = self.df[
                (self.df['datapoint'] == 'Fan Power meter (KW)') |
                (self.df['datapoint'].str.contains('Power', case=False, na=False))
            ].copy()
            
            if power_df.empty:
                return {"error": "No power consumption data found"}
            
            power_df['monitoring_data'] = pd.to_numeric(power_df['monitoring_data'], errors='coerce')
            
            current_avg = power_df['monitoring_data'].mean()
            target_avg = current_avg * (1 - target_reduction_percent / 100)
            reduction_needed = current_avg - target_avg
            
            recommendations = []
            
            # Analyze fan speed
            fan_speed_df = self.df[self.df['datapoint'].str.contains('Fan Speed', case=False, na=False)].copy()
            if not fan_speed_df.empty:
                fan_speed_df['monitoring_data'] = pd.to_numeric(fan_speed_df['monitoring_data'], errors='coerce')
                avg_fan_speed = fan_speed_df['monitoring_data'].mean()
                if avg_fan_speed > 70:
                    recommendations.append({
                        "action": "Reduce fan speed during off-peak hours",
                        "current_value": f"{avg_fan_speed:.1f}%",
                        "target_value": f"{avg_fan_speed * 0.9:.1f}%",
                        "estimated_savings": "2-3%"
                    })
            
            # Analyze temperature setpoints
            temp_df = self.df[self.df['datapoint'].str.contains('temperature setpoint', case=False, na=False)].copy()
            if not temp_df.empty:
                temp_df['monitoring_data'] = pd.to_numeric(temp_df['monitoring_data'], errors='coerce')
                avg_temp = temp_df['monitoring_data'].mean()
                recommendations.append({
                    "action": "Optimize temperature setpoint",
                    "current_value": f"{avg_temp:.1f}°C",
                    "target_value": f"{avg_temp + 1:.1f}°C (increase by 1°C in cooling season)",
                    "estimated_savings": "1-2%"
                })
            
            # Analyze pressure setpoints
            pressure_df = self.df[self.df['datapoint'].str.contains('Pressure setpoint', case=False, na=False)].copy()
            if not pressure_df.empty:
                pressure_df['monitoring_data'] = pd.to_numeric(pressure_df['monitoring_data'], errors='coerce')
                avg_pressure = pressure_df['monitoring_data'].mean()
                recommendations.append({
                    "action": "Reduce supply air pressure setpoint",
                    "current_value": f"{avg_pressure:.0f} Pa",
                    "target_value": f"{avg_pressure * 0.95:.0f} Pa",
                    "estimated_savings": "1-2%"
                })
            
            # Schedule-based recommendations
            recommendations.append({
                "action": "Implement demand-based ventilation schedule",
                "current_value": "Always on",
                "target_value": "Occupancy-based control",
                "estimated_savings": "3-5%"
            })
            
            # Damper control
            damper_df = self.df[self.df['datapoint'].str.contains('damper', case=False, na=False)].copy()
            if not damper_df.empty:
                recommendations.append({
                    "action": "Optimize outdoor air damper control",
                    "current_value": "Standard operation",
                    "target_value": "Economizer cycle when possible",
                    "estimated_savings": "2-4%"
                })
            
            return {
                "current_consumption_kw": float(current_avg),
                "target_consumption_kw": float(target_avg),
                "reduction_needed_kw": float(reduction_needed),
                "target_reduction_percent": target_reduction_percent,
                "recommendations": recommendations,
                "total_estimated_savings": "5-10%"
            }
            
        except Exception as e:
            log.error(f"Error in get_reduction_recommendations: {e}")
            return {"error": str(e)}
    
    def get_consumption_trends(self) -> Dict[str, Any]:
        """Analyze consumption trends over time"""
        try:
            if self.df.empty:
                return {"error": "No data available"}
            
            power_df = self.df[
                (self.df['datapoint'] == 'Fan Power meter (KW)') |
                (self.df['datapoint'].str.contains('Power', case=False, na=False))
            ].copy()
            
            if power_df.empty:
                return {"error": "No power consumption data found"}
            
            power_df['monitoring_data'] = pd.to_numeric(power_df['monitoring_data'], errors='coerce')
            
            if 'data_received_on' in power_df.columns:
                power_df['data_received_on'] = pd.to_datetime(power_df['data_received_on'])
                power_df['hour'] = power_df['data_received_on'].dt.hour
                power_df['day_of_week'] = power_df['data_received_on'].dt.day_name()
                
                # Hourly trends
                hourly_avg = power_df.groupby('hour')['monitoring_data'].mean().round(2)
                peak_hour = hourly_avg.idxmax()
                off_peak_hour = hourly_avg.idxmin()
                
                # Daily trends
                daily_avg = power_df.groupby('day_of_week')['monitoring_data'].mean().round(2)
                
                return {
                    "hourly_average": hourly_avg.to_dict(),
                    "peak_hour": int(peak_hour),
                    "peak_hour_consumption": float(hourly_avg[peak_hour]),
                    "off_peak_hour": int(off_peak_hour),
                    "off_peak_consumption": float(hourly_avg[off_peak_hour]),
                    "daily_average": daily_avg.to_dict(),
                    "overall_average": float(power_df['monitoring_data'].mean()),
                    "trend": "increasing" if power_df['monitoring_data'].iloc[-10:].mean() > power_df['monitoring_data'].iloc[:10].mean() else "decreasing"
                }
            
            return {"error": "No timestamp data available for trend analysis"}
            
        except Exception as e:
            log.error(f"Error in get_consumption_trends: {e}")
            return {"error": str(e)}
    
    def get_equipment_analysis(self) -> Dict[str, Any]:
        """Analyze equipment performance"""
        try:
            if self.df.empty:
                return {"error": "No data available"}
            
            equipment_data = {}
            
            # Analyze fan speed
            fan_df = self.df[self.df['datapoint'].str.contains('Fan Speed', case=False, na=False)].copy()
            if not fan_df.empty:
                fan_df['monitoring_data'] = pd.to_numeric(fan_df['monitoring_data'], errors='coerce')
                equipment_data['fan_speed'] = {
                    "average": float(fan_df['monitoring_data'].mean()),
                    "max": float(fan_df['monitoring_data'].max()),
                    "min": float(fan_df['monitoring_data'].min()),
                    "status": "normal" if 30 <= fan_df['monitoring_data'].mean() <= 80 else "needs_attention"
                }
            
            # Analyze temperature
            temp_df = self.df[self.df['datapoint'].str.contains('Temp', case=False, na=False)].copy()
            if not temp_df.empty:
                temp_df['monitoring_data'] = pd.to_numeric(temp_df['monitoring_data'], errors='coerce')
                equipment_data['temperature'] = {
                    "average": float(temp_df['monitoring_data'].mean()),
                    "max": float(temp_df['monitoring_data'].max()),
                    "min": float(temp_df['monitoring_data'].min()),
                    "status": "normal" if 20 <= temp_df['monitoring_data'].mean() <= 26 else "out_of_range"
                }
            
            # Analyze pressure
            pressure_df = self.df[self.df['datapoint'].str.contains('pressure', case=False, na=False)].copy()
            if not pressure_df.empty:
                pressure_df['monitoring_data'] = pd.to_numeric(pressure_df['monitoring_data'], errors='coerce')
                equipment_data['pressure'] = {
                    "average": float(pressure_df['monitoring_data'].mean()),
                    "max": float(pressure_df['monitoring_data'].max()),
                    "min": float(pressure_df['monitoring_data'].min())
                }
            
            if not equipment_data:
                return {"error": "No equipment data found"}
            
            return equipment_data
            
        except Exception as e:
            log.error(f"Error in get_equipment_analysis: {e}")
            return {"error": str(e)}
    
    def get_cost_analysis(self, cost_per_kwh: float = 0.15) -> Dict[str, Any]:
        """Analyze energy costs"""
        try:
            if self.df.empty:
                return {"error": "No data available"}
            
            power_df = self.df[
                (self.df['datapoint'] == 'Fan Power meter (KW)') |
                (self.df['datapoint'].str.contains('Power', case=False, na=False))
            ].copy()
            
            if power_df.empty:
                return {"error": "No power consumption data found"}
            
            power_df['monitoring_data'] = pd.to_numeric(power_df['monitoring_data'], errors='coerce')
            
            total_kwh = power_df['monitoring_data'].sum()
            avg_kw = power_df['monitoring_data'].mean()
            
            # Estimate daily, monthly, yearly costs
            hours_of_data = len(power_df)
            daily_cost = avg_kw * 24 * cost_per_kwh
            monthly_cost = daily_cost * 30
            yearly_cost = monthly_cost * 12
            
            # Cost by area if available
            cost_by_area = {}
            if 'site' in power_df.columns:
                for site in power_df['site'].unique():
                    site_data = power_df[power_df['site'] == site]
                    site_avg = site_data['monitoring_data'].mean()
                    site_daily_cost = site_avg * 24 * cost_per_kwh
                    cost_by_area[site] = {
                        "daily_cost": float(site_daily_cost),
                        "monthly_cost": float(site_daily_cost * 30),
                        "yearly_cost": float(site_daily_cost * 365)
                    }
            
            return {
                "cost_per_kwh": cost_per_kwh,
                "current_avg_kw": float(avg_kw),
                "estimated_daily_cost": float(daily_cost),
                "estimated_monthly_cost": float(monthly_cost),
                "estimated_yearly_cost": float(yearly_cost),
                "cost_by_area": cost_by_area,
                "potential_savings_5_percent": {
                    "daily": float(daily_cost * 0.05),
                    "monthly": float(monthly_cost * 0.05),
                    "yearly": float(yearly_cost * 0.05)
                }
            }
            
        except Exception as e:
            log.error(f"Error in get_cost_analysis: {e}")
            return {"error": str(e)}
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the system"""
        try:
            if self.df.empty:
                return {"error": "No data available"}
            
            summary = {
                "total_datapoints": len(self.df),
                "unique_equipment": self.df['equipment_name'].nunique() if 'equipment_name' in self.df.columns else 0,
                "unique_sites": self.df['site'].nunique() if 'site' in self.df.columns else 0,
                "data_types": self.df['datapoint'].unique().tolist() if 'datapoint' in self.df.columns else [],
            }
            
            # Add energy summary
            energy_data = self.get_energy_by_area()
            if "error" not in energy_data:
                summary["energy_summary"] = energy_data
            
            # Add equipment summary
            equipment_data = self.get_equipment_analysis()
            if "error" not in equipment_data:
                summary["equipment_summary"] = equipment_data
            
            return summary
            
        except Exception as e:
            log.error(f"Error in get_overall_summary: {e}")
            return {"error": str(e)}


class QuestionMatcher:
    """No longer needed - ChatGPT handles all question understanding"""
    pass


def generate_chatgpt_response(question: str, analysis_data: Dict[str, Any], context: Dict[str, Any] = None) -> ChatResponse:
    """Use ChatGPT to generate intelligent responses based on analysis data"""
    
    if not client:
        # Fallback to simple response if OpenAI is not available
        return ChatResponse(
            answer="OpenAI API is not configured. Please set OPENAI_API_KEY environment variable.",
            data={},
            recommendations=[],
            confidence=0.0
        )
    
    try:
        # Prepare the system prompt with data context
        system_prompt = f"""You are an expert BMS (Building Management System) energy analyst assistant. 
Your role is to help users understand their energy consumption patterns and provide actionable recommendations 
to optimize energy usage and reduce costs.

You have access to the following analyzed data from the BMS:
{json.dumps(analysis_data, indent=2)}

Guidelines:
1. Provide clear, concise answers based on the data
2. Use specific numbers and percentages from the data
3. Format important values in **bold**
4. Give actionable, practical recommendations
5. Be friendly and professional
6. If the data shows an error, acknowledge it and explain what data you need
7. Focus on energy efficiency and cost savings
8. Highlight the most impactful insights first
"""

        user_prompt = f"""User Question: {question}

Please analyze the data and provide:
1. A clear answer to the question
2. Key insights from the data
3. Actionable recommendations (3-5 items)
4. Your confidence level in this answer (0-1 scale)

Format your response as JSON with this structure:
{{
    "answer": "Your detailed answer here with **bold** for important values",
    "key_insights": ["insight 1", "insight 2", ...],
    "recommendations": ["recommendation 1", "recommendation 2", ...],
    "confidence": 0.85
}}
"""

        # Call ChatGPT
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        gpt_response = json.loads(response.choices[0].message.content)
        
        return ChatResponse(
            answer=gpt_response.get("answer", "I couldn't generate a proper answer."),
            data=analysis_data,
            recommendations=gpt_response.get("recommendations", []),
            confidence=float(gpt_response.get("confidence", 0.5))
        )
        
    except Exception as e:
        log.error(f"ChatGPT error: {e}")
        return ChatResponse(
            answer=f"I encountered an error while processing your question: {str(e)}",
            data=analysis_data,
            recommendations=[],
            confidence=0.0
        )


def analyze_question_intent(question: str) -> str:
    """Use ChatGPT to determine what type of analysis to perform"""
    
    if not client:
        # Fallback to simple keyword matching
        question_lower = question.lower()
        if any(word in question_lower for word in ["which", "most", "highest", "area", "where"]):
            return "energy_by_area"
        elif any(word in question_lower for word in ["reduce", "save", "decrease", "lower", "optimize"]):
            return "reduce_energy"
        else:
            return "general"
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": """Analyze the user's question and determine what type of BMS energy analysis they need.
                    
Available analysis types:
- energy_by_area: Questions about which area/site/equipment consumes most energy
- reduce_energy: Questions about reducing/optimizing energy consumption
- trend_analysis: Questions about historical patterns or trends
- equipment_health: Questions about equipment performance or health
- cost_analysis: Questions about energy costs or spending
- general: General questions about the system

Respond with ONLY the analysis type, nothing else."""
                },
                {"role": "user", "content": question}
            ],
            # temperature=0.3,
            max_tokens=100
        )
        
        intent = response.choices[0].message.content.strip().lower()
        log.info(f"Detected intent: {intent}")
        print(f"Detected intent: {intent}")
        return intent
        
    except Exception as e:
        log.error(f"Intent detection error: {e}")
        return "general"


@router.post("/ask", response_model=ChatResponse)
def ask_chatbot(request: ChatRequest):
    """
    Ask the chatbot ANY question about energy consumption.
    
    Powered by ChatGPT for natural language understanding.
    
    Example questions:
    - "Which area consumes the most electricity?"
    - "How can I reduce energy by 10%?"
    - "What's the average power consumption in the last week?"
    - "Show me the top 3 energy consumers"
    - "Is my HVAC system running efficiently?"
    - "What time of day uses the most energy?"
    """
    try:
        log.info(f"Received question: {request.question}")
        
        # Use ChatGPT to understand the question intent
        intent = analyze_question_intent(request.question)
        log.info(f"Detected intent: {intent}")
        
        # Initialize analyzer with data
        analyzer = EnergyAnalyzer(request.data)
        
        # Perform appropriate analysis based on intent
        analysis_result = {}
        
        if intent == "energy_by_area":
            analysis_result = analyzer.get_energy_by_area()
        elif intent == "reduce_energy":
            # Extract reduction target using regex
            import re
            match = re.search(r'(\d+(?:\.\d+)?)\s*%|by\s+(\d+(?:\.\d+)?)', request.question.lower())
            target_percent = float(match.group(1) or match.group(2)) if match else 5.0
            analysis_result = analyzer.get_reduction_recommendations(target_percent)
        elif intent == "trend_analysis":
            analysis_result = analyzer.get_consumption_trends()
        elif intent == "equipment_health":
            analysis_result = analyzer.get_equipment_analysis()
        elif intent == "cost_analysis":
            analysis_result = analyzer.get_cost_analysis()
        else:
            # For general questions, provide overall summary
            analysis_result = analyzer.get_overall_summary()
        
        # Use ChatGPT to generate natural, intelligent response
        response = generate_chatgpt_response(request.question, analysis_result, request.context)
        
        log.info(f"Response confidence: {response.confidence}")
        return response
        
    except Exception as e:
        log.error(f"Chatbot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-questions")
def get_supported_questions():
    """Get list of supported question types"""
    return {
        "supported_questions": [
            {
                "category": "Energy Analysis",
                "examples": [
                    "Which area uses the most electricity?",
                    "Where is most energy being consumed?",
                    "Which site has the highest consumption?"
                ]
            },
            {
                "category": "Energy Optimization",
                "examples": [
                    "How can I reduce energy consumption by 5%?",
                    "What should I do to save energy next month?",
                    "How do I decrease power consumption by 10%?"
                ]
            }
        ]
    }
