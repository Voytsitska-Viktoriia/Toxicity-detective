import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Loader2, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { pipeline } from "@huggingface/transformers";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";

const categories = [
  { name: 'Toxicity', color: 'bg-red-500' },
];

const toxicPhrasePatterns = {
  toxicity: [
    "hate", "idiot", "stupid", "pathetic", "dumb", "moron", "terrible", 
    "useless", "disgusting", "awful", "horrible", "trash", "garbage", "worst",
    "loser", "worthless", "scum", "failure", "retard", "boring", "lame"
  ],
};

const contextualToxicityIndicators = {
  intensifiers: ["fucking", "absolutely", "totally", "completely", "utterly", "so", "such", "really"],
  negativeActions: ["hate", "kill", "destroy", "hurt", "punish", "slap", "hit", "ruin", "break"],
  targetsOfAbuse: ["you", "they", "people", "everyone", "group", "community", "him", "her", "them"]
};

const positivePhrasesExamples = [
  "good job", "well done", "congratulations", "thank you", 
  "great work", "nice", "awesome", "excellent", "amazing",
  "appreciate", "thanks", "good", "wonderful", "fantastic",
  "love", "happy", "pleased", "superb", "impressive", "bravo",
  "helpful", "kind", "thoughtful", "brilliant", "outstanding"
];
let classifier = null

const ToxicityAnalyzer = () => {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);
  const { toast } = useToast();
  const [modelInfo, setModelInfo] = useState('');

  useEffect(() => {
    async function loadModel() {
      try {
        setIsModelLoading(true);
        
        const toxicityClassifier = await pipeline(
          'text-classification',
          'Xenova/toxic-bert',
          { 
            revision: 'main',
            quantized: false
          }
        );
        console.log(toxicityClassifier)
        
        classifier = toxicityClassifier 
        setModelError(null);
        setModelInfo('Using ToxicBERT classification model for toxicity detection');
        console.log("Toxicity model loaded successfully");
      } catch (error) {
        console.error("Error loading toxicity model:", error);
        setModelError("Failed to load toxicity detection model. Using fallback method.");
        setModelInfo('Using enhanced rule-based fallback detection (Jigsaw dataset patterns)');
      } finally {
        setIsModelLoading(false);
      }
    }

    loadModel();
  }, []);

  const fallbackAnalyzeSentiment = (inputText: string) => {
    const lowerText = inputText.toLowerCase().trim();
    const words = lowerText.split(/\s+/);
    const wordCount = words.length;
    
    let scores = {
      toxicity: 0,
    };
    
    let positiveScore = 0;
    positivePhrasesExamples.forEach(phrase => {
      if (lowerText.includes(phrase)) {
        positiveScore += 1;
      }
    });
    
    Object.entries(toxicPhrasePatterns).forEach(([category, patterns]) => {
      patterns.forEach(pattern => {
        if (lowerText.includes(pattern)) {
          scores[category as keyof typeof scores] += 1;
        }
      });
    });
    
    let contextualToxicityScore = 0;
    for (let i = 0; i < words.length - 1; i++) {
      if (contextualToxicityIndicators.intensifiers.some(intensifier => words[i].includes(intensifier)) && 
          Object.values(toxicPhrasePatterns).flat().some(toxic => words[i+1].includes(toxic))) {
        contextualToxicityScore += 1;
        scores.toxicity += 0.5;
      }
    
      if (contextualToxicityIndicators.negativeActions.some(action => words[i].includes(action)) && 
          contextualToxicityIndicators.targetsOfAbuse.some(target => words[i+1].includes(target))) {
        contextualToxicityScore += 1;
      }
    }
    
    const normalizeScore = (score: number) => Math.min(1, Math.max(0, score / (wordCount * 0.25)));
    
    if (positiveScore > (scores.toxicity) * 0.7) {
      return categories.map(category => {
        const key = category.name.toLowerCase() as keyof typeof scores;
        let baseScore = Math.max(0, 0.05 - (positiveScore * 0.01));
        return { 
          category: category.name, 
          score: key in scores ? baseScore : baseScore * 0.2 
        };
      });
    }
    
    return categories.map(category => {
      const key = category.name.toLowerCase() as keyof typeof scores;
      return { 
        category: category.name, 
        score: normalizeScore(scores[key]) 
      };
    });
  };

  const mapToxicBertToCategories = (prediction: any, inputText:string) => {
    try {
      console.log("Raw model prediction:", prediction);
      
      const isToxic = prediction.some((p: any) => 
        (p.label.includes('toxic') && p.score > 0.5));
      
      if (isToxic) {
        const toxicPrediction = prediction.find((p: any) => 
          p.label.includes('toxic'));
        
        return [
          { 
            category: 'Toxicity', 
            score: toxicPrediction ? toxicPrediction.score : 0.2 
          },  
        ];
      } else {
        return categories.map(category => ({
          category: category.name,
          score: 0.00 
        }));
      }
    } catch (error) {
      console.error("Error mapping model results:", error);
      return fallbackAnalyzeSentiment(inputText);
    }
  };

  const analyzeWithModel = async (inputText: string) => {
    try {
      const prediction = await classifier(inputText);
      console.log("Model prediction:", prediction);
      
      return mapToxicBertToCategories(prediction, inputText);
    } catch (error) {
      console.error("Error using the model for prediction:", error);
      return fallbackAnalyzeSentiment(inputText);
    }
  };

  const analyzeText = async () => {
    if (!text.trim()) return;
    
    setIsAnalyzing(true);
    
    try {
      let analysisResults;
      
      if (classifier && !modelError) {
        analysisResults = await analyzeWithModel(text);
      } else {
        analysisResults = fallbackAnalyzeSentiment(text);
      }
      
      setResults(analysisResults);
      
      toast({
        title: "Analysis complete",
        description: "Text has been analyzed successfully",
      });
    } catch (error) {
      console.error('Error analyzing text:', error);
      
      toast({
        variant: "destructive",
        title: "Analysis failed",
        description: "An error occurred while analyzing the text",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 p-6 sm:p-8">
      <div className="max-w-3xl mx-auto space-y-8">
        <div className="space-y-2 text-center">
          <Badge variant="secondary" className="mb-4">
            Content Analysis
          </Badge>
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl">
            Toxicity Detective
          </h1>
          <p className="text-gray-500 max-w-lg mx-auto">
            Analyze text for signs of toxicity
          </p>
        </div>

        <Card className="p-6 backdrop-blur-sm bg-white/80 border border-gray-200">
          <div className="space-y-4">
            <Textarea
              placeholder="Enter text to analyze..."
              className="min-h-[120px] resize-none"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            
            {modelInfo && (
              <Alert variant="default" className="bg-blue-50 border-blue-200">
                <AlertTitle className="text-blue-700">Model Information</AlertTitle>
                <AlertDescription className="text-blue-600">
                  {modelInfo}
                </AlertDescription>
              </Alert>
            )}
            
            {modelError && (
              <div className="flex items-center gap-2 text-amber-600 bg-amber-50 p-3 rounded-md text-sm">
                <AlertCircle className="h-4 w-4" />
                <span>{modelError}</span>
              </div>
            )}
            
            {isModelLoading && (
              <div className="flex items-center gap-2 text-blue-600 bg-blue-50 p-3 rounded-md text-sm">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Loading toxicity detection model...</span>
              </div>
            )}
            
            <Button
              className="w-full relative"
              onClick={analyzeText}
              disabled={isAnalyzing || !text.trim()}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing
                </>
              ) : (
                'Analyze Text'
              )}
            </Button>
          </div>
        </Card>

        {results && (
          <Card className="p-6 backdrop-blur-sm bg-white/80 border border-gray-200 animate-fade-in">
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-900">Analysis Results</h2>
              <div className="space-y-3">
                {categories.map((category, index) => {
                  const resultItem = results.find((r: any) => r.category === category.name);
                  const score = resultItem ? Math.round(resultItem.score * 100) : 0;
                  
                  return (
                    <div key={category.name} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">{category.name}</span>
                        <span className="text-gray-900 font-medium">
                          {score}%
                        </span>
                      </div>
                      <Progress
                        value={score}
                        className={`h-2 ${category.color}`}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

export default ToxicityAnalyzer;
