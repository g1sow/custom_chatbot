from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="G1SOW Studio RAG Chatbot API", version="1.0.0")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client with error handling
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

# Website knowledge base
KNOWLEDGE_BASE = [
    {
        "content": "G1SOW is a digital agency founded in 2025 that specializes in creating exceptional web experiences. We have completed 10+ projects with 98% client satisfaction, have 5 team members, and 2+ years of experience.",
        "category": "about",
        "keywords": ["about", "company", "team", "experience", "founded"]
    },
    {
        "content": "Our AI-powered services include: AI-Powered Websites & E-commerce - Intelligent sites that adapt to users, delivering personalized experiences that boost engagement and conversion. AI Chatbot Integration - Smart conversational agents that learn from interactions to provide 24/7 customer support. AI Calling Agents - Voice assistants that handle calls, appointments, and inquiries with natural conversation. Predictive Analytics - Data visualization tools that forecast trends and uncover opportunities. Advanced AI Agents - Custom AI solutions that automate complex tasks. Startup AI Solutions - Tailored AI tools for startups that accelerate growth.",
        "category": "services",
        "keywords": ["services", "AI", "chatbot", "e-commerce", "analytics", "startup", "automation"]
    },
    {
        "content": "Our featured projects include: Homly State - A real estate platform with advanced search filters. Hayes Valley - Interior design site with online booking system and consultation. Susmis Nail Zone - Brand website for nail art services. Restaurant - Brand website with food ordering system.",
        "category": "projects",
        "keywords": ["projects", "portfolio", "real estate", "interior design", "restaurant", "nail zone"]
    },
    {
        "content": "Client testimonials: Susmi from Susmis Nail Zone said: 'Working with G1SOW was great. Their AI chatbot integration and automated customer service tools have revolutionized how we interact with clients.' Harry Smith from cafe50 said: 'These folks have incredible attention to detail. The AI analytics dashboard they implemented helps us understand customer behavior better than ever.'",
        "category": "reviews",
        "keywords": ["reviews", "testimonials", "clients", "feedback", "satisfaction"]
    },
    {
        "content": "Our design process begins with understanding your goals and target audience. We then create wireframes, followed by design mockups. After your approval, we move into development, testing, and finally launch. We maintain open communication throughout the entire process.",
        "category": "process",
        "keywords": ["process", "design", "development", "wireframes", "testing", "launch"]
    },
    {
        "content": "FAQ: What services does g1sow offer? We offer web development, AI integration, e-commerce solutions, CMS integration, SEO optimization, and ongoing maintenance. How long does a project take? Basic websites take 4-6 weeks, complex applications take 3-6 months. Do you provide hosting and maintenance? Yes, we offer both hosting and maintenance services including security monitoring and performance optimization.",
        "category": "faq",
        "keywords": ["faq", "questions", "timeline", "hosting", "maintenance", "pricing"]
    },
    {
        "content": "We implement SSL certificates, regular security updates, firewall protection, and follow best practices for secure coding. We also offer ongoing security monitoring for all our clients. Our pricing is project-based and depends on scope, complexity, and timeline.",
        "category": "technical",
        "keywords": ["security", "pricing", "SSL", "firewall", "monitoring"]
    }
]

# Initialize TF-IDF vectorizer
try:
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    knowledge_texts = [item["content"] for item in KNOWLEDGE_BASE]
    tfidf_matrix = vectorizer.fit_transform(knowledge_texts)
    logger.info("TF-IDF vectorizer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize TF-IDF vectorizer: {str(e)}")
    vectorizer = None
    tfidf_matrix = None

def find_relevant_context(query: str, top_k: int = 3) -> List[Dict]:
    """Find most relevant knowledge base entries for the query"""
    if not vectorizer or tfidf_matrix is None:
        return []
    
    try:
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_contexts = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                relevant_contexts.append({
                    "content": KNOWLEDGE_BASE[idx]["content"],
                    "category": KNOWLEDGE_BASE[idx]["category"],
                    "similarity": float(similarities[idx])
                })
        
        return relevant_contexts
    except Exception as e:
        logger.error(f"Error finding relevant context: {str(e)}")
        return []

def generate_response(query: str, contexts: List[Dict]) -> str:
    """Generate response using Groq with retrieved contexts"""
    if not client:
        return "I apologize, but the AI service is currently unavailable. Please contact our team directly for assistance with G1SOW Studio's services."
    
    context_text = "\n\n".join([ctx["content"] for ctx in contexts])
    
    system_prompt = """You are a helpful assistant for G1SOW Studio, a digital agency specializing in AI-powered web development. 
    You should only answer questions based on the provided context about G1SOW Studio's services, projects, team, and processes.
    
    Guidelines:
    - Be friendly and professional
    - Only provide information that's in the context
    - If you don't have enough information, politely say so and suggest contacting the team
    - Keep responses concise but informative
    - Always stay focused on G1SOW Studio's offerings
    
    Context about G1SOW Studio:
    {context}
    
    Please answer the user's question based only on this context."""
    
    user_prompt = f"Question: {query}"
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt.format(context=context_text)},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500,
            top_p=0.9
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again or contact our team directly."

@app.get("/")
async def root():
    return {"message": "G1SOW Studio RAG Chatbot API is running!", "status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        relevant_contexts = find_relevant_context(request.message)
        
        if not relevant_contexts:
            response_text = "I don't have specific information about that topic. For detailed information about G1SOW Studio's services, please visit our website or contact our team directly."
            sources = []
        else:
            response_text = generate_response(request.message, relevant_contexts)
            sources = [ctx["category"] for ctx in relevant_contexts]
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            conversation_id=request.conversation_id or "default"
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "G1SOW Studio RAG Chatbot",
        "groq_client": "connected" if client else "disconnected",
        "vectorizer": "initialized" if vectorizer else "failed"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
