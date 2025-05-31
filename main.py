from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Dict
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="G1SOW Studio Smart Sales Assistant", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
try:
    from groq import Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        client = None
    else:
        client = Groq(api_key=groq_api_key)
        logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {str(e)}")
    client = None

class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    conversation_id: str
    intent: str = "general"

# Enhanced knowledge base with sales-focused content
KNOWLEDGE_BASE = [
    {
        "content": "G1SOW Studio is a cutting-edge digital agency founded in 2025, specializing in AI-powered web experiences. We've completed 10+ successful projects with 98% client satisfaction. Our 5-member expert team brings 2+ years of specialized AI development experience. We're not just developers - we're digital transformation partners who help businesses leverage AI to increase revenue, reduce costs, and stay ahead of competition.",
        "category": "about",
        "keywords": ["about", "company", "team", "experience", "founded", "who are you", "tell me about"],
        "sales_angle": "expertise_credibility"
    },
    {
        "content": "Our AI-powered services revolutionize how businesses operate online: 🤖 AI-Powered Websites & E-commerce - Intelligent sites that adapt to users, boosting engagement by 40% and conversions by 25%. 💬 AI Chatbot Integration - Smart conversational agents providing 24/7 customer support, reducing response time by 90%. 📞 AI Calling Agents - Voice assistants handling calls and appointments, saving 15+ hours weekly. 📊 Predictive Analytics - Data visualization tools forecasting trends and uncovering hidden opportunities. 🚀 Advanced AI Agents - Custom automation solutions that eliminate repetitive tasks. 💡 Startup AI Solutions - Tailored AI tools that accelerate growth and reduce operational costs by 30%.",
        "category": "services",
        "keywords": ["services", "AI", "chatbot", "e-commerce", "analytics", "startup", "automation", "what do you do", "offerings"],
        "sales_angle": "value_proposition"
    },
    {
        "content": "Our portfolio showcases real results: 🏠 Homly State - Real estate platform with AI-powered search, increasing property inquiries by 60%. 🎨 Hayes Valley - Interior design site with smart booking system, boosting consultation bookings by 45%. 💅 Susmis Nail Zone - Brand website with AI customer service, improving client retention by 35%. 🍽️ Restaurant Solutions - Food ordering systems with predictive analytics, increasing average order value by 20%.",
        "category": "projects",
        "keywords": ["projects", "portfolio", "real estate", "interior design", "restaurant", "nail zone", "examples", "case studies"],
        "sales_angle": "social_proof"
    },
    {
        "content": "Client success stories that speak volumes: 💬 Susmi (Susmis Nail Zone): 'G1SOW's AI chatbot integration revolutionized our client interactions. We now handle 3x more inquiries without additional staff!' ⭐ Harry Smith (Cafe50): 'Their AI analytics dashboard helped us understand customer behavior patterns we never knew existed. Revenue increased 25% in just 3 months!' These aren't just testimonials - they're proof of ROI our clients achieve.",
        "category": "reviews",
        "keywords": ["reviews", "testimonials", "clients", "feedback", "satisfaction", "success stories"],
        "sales_angle": "social_proof"
    },
    {
        "content": "Our proven development process ensures your success: 1️⃣ Discovery & Strategy - We dive deep into your business goals and target audience. 2️⃣ AI-Powered Design - Create wireframes and mockups with user behavior predictions. 3️⃣ Smart Development - Build with cutting-edge AI integrations. 4️⃣ Testing & Optimization - Rigorous testing with performance analytics. 5️⃣ Launch & Growth - Deploy with ongoing AI-driven optimizations. We maintain transparent communication and provide weekly progress updates.",
        "category": "process",
        "keywords": ["process", "design", "development", "wireframes", "testing", "launch", "how do you work"],
        "sales_angle": "process_confidence"
    },
    {
        "content": "Frequently asked questions with honest answers: ❓ What makes G1SOW different? We're AI-first, not AI-added. Every solution is built with intelligence from the ground up. ⏱️ Project timelines? Basic AI websites: 4-6 weeks. Complex AI applications: 3-6 months. We never compromise quality for speed. 🛡️ Hosting & maintenance? Yes! We provide secure hosting, 24/7 monitoring, and proactive maintenance. Your success is our ongoing commitment. 💰 Investment? Project-based pricing starting from $5,000 for basic AI websites to $50,000+ for enterprise solutions. ROI typically achieved within 6 months.",
        "category": "faq",
        "keywords": ["faq", "questions", "timeline", "hosting", "maintenance", "pricing", "cost", "how much"],
        "sales_angle": "objection_handling"
    },
    {
        "content": "Security and technical excellence you can trust: 🔒 Enterprise-grade SSL certificates and encryption. 🛡️ Regular security updates and vulnerability assessments. 🔥 Advanced firewall protection and DDoS mitigation. 👨‍💻 Secure coding practices following OWASP guidelines. 📊 Real-time security monitoring and instant threat response. 🚀 99.9% uptime guarantee with performance optimization. We don't just build websites - we build digital fortresses for your business.",
        "category": "technical",
        "keywords": ["security", "SSL", "firewall", "monitoring", "technical", "uptime", "performance"],
        "sales_angle": "trust_building"
    },
    {
        "content": "Ready to transform your business with AI? 🚀 Free consultation available - Let's discuss your vision and show you exactly how AI can boost your revenue. 📞 Call us directly or schedule a video call to see live demos. 💡 Custom proposals within 48 hours with detailed ROI projections. 🎯 No obligation, just honest advice about how we can help you dominate your market with AI. Contact G1SOW Studio today - your competitors are already exploring AI, don't let them get ahead!",
        "category": "cta",
        "keywords": ["contact", "consultation", "free", "demo", "proposal", "call", "schedule", "get started"],
        "sales_angle": "urgency_cta"
    }
]

# Initialize enhanced vectorizer
try:
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=2000,
        ngram_range=(1, 3),  # Include phrases
        min_df=1,
        max_df=0.95
    )
    knowledge_texts = [item["content"] + " " + " ".join(item["keywords"]) for item in KNOWLEDGE_BASE]
    tfidf_matrix = vectorizer.fit_transform(knowledge_texts)
    logger.info("Enhanced TF-IDF vectorizer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize TF-IDF vectorizer: {str(e)}")
    vectorizer = None
    tfidf_matrix = None

def detect_intent(query: str) -> str:
    """Detect user intent from the query"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['price', 'cost', 'budget', 'expensive', 'cheap', 'afford']):
        return 'pricing'
    elif any(word in query_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
        return 'greeting'
    elif any(word in query_lower for word in ['help', 'support', 'problem', 'issue', 'trouble']):
        return 'support'
    elif any(word in query_lower for word in ['services', 'what do you do', 'offerings', 'capabilities']):
        return 'services'
    elif any(word in query_lower for word in ['portfolio', 'projects', 'examples', 'case studies']):
        return 'portfolio'
    elif any(word in query_lower for word in ['contact', 'call', 'email', 'reach', 'consultation']):
        return 'contact'
    elif any(word in query_lower for word in ['about', 'who are you', 'company', 'team']):
        return 'about'
    else:
        return 'general'

def find_relevant_context(query: str, top_k: int = 3) -> List[Dict]:
    """Enhanced context retrieval with intent awareness"""
    if not vectorizer or tfidf_matrix is None:
        return []
    
    try:
        # Expand query with synonyms for better matching
        expanded_query = query + " " + " ".join([
            "AI artificial intelligence",
            "website web development",
            "chatbot automation",
            "digital agency services"
        ])
        
        query_vector = vectorizer.transform([expanded_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get top-k most similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_contexts = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Lower threshold for better coverage
                relevant_contexts.append({
                    "content": KNOWLEDGE_BASE[idx]["content"],
                    "category": KNOWLEDGE_BASE[idx]["category"],
                    "sales_angle": KNOWLEDGE_BASE[idx]["sales_angle"],
                    "similarity": float(similarities[idx])
                })
        
        return relevant_contexts
    except Exception as e:
        logger.error(f"Error finding relevant context: {str(e)}")
        return []

def generate_sales_response(query: str, contexts: List[Dict], intent: str) -> str:
    """Generate intelligent sales-focused response"""
    if not client:
        return "Hi there! I'm Sarah, your AI assistant from G1SOW Studio. While our main AI system is updating, I can still help you! We specialize in AI-powered websites and automation that boost business revenue. What specific challenge can we help you solve today? 🚀"
    
    context_text = "\n\n".join([ctx["content"] for ctx in contexts])
    
    # Enhanced system prompt with sales personality
    system_prompt = f"""You are Sarah, a friendly and knowledgeable sales consultant for G1SOW Studio, a premium AI-powered web development agency. You're professional yet conversational, confident but not pushy.

PERSONALITY TRAITS:
- Enthusiastic about AI and its business benefits
- Solution-oriented and consultative
- Uses emojis strategically (1-2 per response)
- Focuses on value and ROI, not just features
- Creates curiosity and urgency without being aggressive
- Always guides toward next steps (consultation, demo, contact)

CONVERSATION RULES:
1. ALWAYS provide helpful information, even for off-topic queries
2. Smoothly connect any topic back to how G1SOW can help their business
3. Use specific numbers and results when available
4. Ask engaging follow-up questions
5. Create FOMO (fear of missing out) subtly
6. End with a soft call-to-action

USER INTENT: {intent}

CONTEXT ABOUT G1SOW STUDIO:
{context_text}

RESPONSE GUIDELINES:
- If greeting: Be warm, introduce yourself, ask about their business needs
- If pricing: Discuss value first, then mention investment ranges with ROI focus
- If services: Highlight unique AI advantages and business benefits
- If off-topic: Acknowledge their question, then bridge to relevant services
- If vague: Ask clarifying questions to understand their business challenges

Remember: You're not just answering questions - you're building relationships and creating desire for G1SOW's services. Every response should leave them wanting to know more!"""

    user_prompt = f"User query: {query}"
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,  # More creative for sales conversations
            max_tokens=800,   # Longer responses for better engagement
            top_p=0.9
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Hi! I'm Sarah from G1SOW Studio 👋 While I'm having a small technical hiccup, I'm excited to help you discover how our AI-powered solutions can transform your business! We've helped clients increase revenue by 25-40% with smart automation. What's your biggest business challenge right now? Let's chat about how we can solve it! 🚀"

@app.get("/")
async def root():
    return {
        "message": "G1SOW Studio Smart Sales Assistant API is running!",
        "version": "2.0.0",
        "features": ["RAG-based responses", "Sales-focused AI", "Intent detection", "Smart context retrieval"]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with sales intelligence"""
    try:
        # Detect user intent
        intent = detect_intent(request.message)
        
        # Find relevant context
        relevant_contexts = find_relevant_context(request.message, top_k=4)
        
        # Generate intelligent sales response
        response_text = generate_sales_response(request.message, relevant_contexts, intent)
        sources = [ctx["category"] for ctx in relevant_contexts]
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            conversation_id=request.conversation_id or "sales-chat",
            intent=intent
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        # Fallback response that's still sales-focused
        fallback_response = "Hi there! I'm Sarah from G1SOW Studio 🌟 Even though I'm experiencing a small technical issue, I'm here to help! We're an AI-first agency that's helped 10+ businesses boost their revenue with smart automation. What brings you here today? Are you looking to increase sales, reduce costs, or stay ahead of competitors with AI? Let's chat! 🚀"
        
        return ChatResponse(
            response=fallback_response,
            sources=["fallback"],
            conversation_id=request.conversation_id or "sales-chat",
            intent="support"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "G1SOW Studio Smart Sales Assistant",
        "version": "2.0.0",
        "groq_client": "connected" if client else "disconnected",
        "vectorizer": "initialized" if vectorizer else "failed"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
