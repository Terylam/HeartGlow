### Frontend Setup:

cd frontend
npm create vite@latest . -- --template react
npm install        
npm install axios react-router-dom
npm install framer-motion
npm install lucide-react #NEW
npm install -D tailwindcss@4.1.18 @tailwindcss/vite@4.1.18
(copy the src.zip)

=================================
### Backend Setup: 
cd backend
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
(copy the main.py, test_rag_evaluation.py, requirements.txt)

#check library list
venv/bin/pip list
=================================

### MongoDB Setup: 
brew install mongodb-community-shell

=================================

check ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull 'model' ; ollama run 'model'
ollama list
ollama ps #model in current use 
---
Cline:
npm install -g cline
