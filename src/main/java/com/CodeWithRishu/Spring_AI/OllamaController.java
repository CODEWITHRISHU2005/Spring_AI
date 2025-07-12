package com.CodeWithRishu.Spring_AI;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/ollama")
@CrossOrigin(origins = "http://localhost:5173")
public class OllamaController {

    private final ChatClient chatClient;
    private final EmbeddingModel embeddingModel;
    private final VectorStore vectorStore;

    @Autowired
    public OllamaController(EmbeddingModel embeddingModel, OllamaChatModel chatModel, VectorStore vectorStore) {
        this.vectorStore = vectorStore;
        this.chatClient = ChatClient.create(chatModel);
        this.embeddingModel = embeddingModel;
    }

    @GetMapping("/{message}")
    public ResponseEntity<String> getAnswer(@PathVariable String message) {

        ChatResponse chatResponse = chatClient
                .prompt(message)
                .call()
                .chatResponse();


        assert chatResponse != null;
        System.out.println(chatResponse.getMetadata().getModel());


        String response = chatResponse
                .getResult()
                .getOutput()
                .getText();

        return ResponseEntity.ok(response);
    }

    @PostMapping("/recommend")
    public ResponseEntity<String> getRecommendation(@RequestParam String type,
                                                    @RequestParam String year,
                                                    @RequestParam String lang) {
        String tempt = """
                I want to watch a {type} movie tonight with good rating,
                looking for movies around this year {year} and language {lang}.
                Suggest me one specific movie and tell me the cast and length of the movie.
                """;
        PromptTemplate promptTemplate = new PromptTemplate(tempt);
        Prompt prompt = promptTemplate.create(Map.of("type", type, "year", year, "lang", lang));

        String response = chatClient
                .prompt(prompt)
                .call()
                .content();
        return ResponseEntity.ok(response);
    }

    @PostMapping("/embedding")
    public float[] getEmbedding(@RequestParam String text) {
        return embeddingModel.embed(text);
    }

    @PostMapping("/similarity")
    public ResponseEntity<Double> getSimilarity(@RequestParam String text1, @RequestParam String text2) {
        float[] embedding1 = embeddingModel.embed(text1);
        float[] embedding2 = embeddingModel.embed(text2);

        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (int i = 0; i < embedding1.length; i++) {
            dotProduct += embedding1[i] * embedding2[i];
            norm1 += Math.pow(embedding1[i], 2);
            norm2 += Math.pow(embedding2[i], 2);
        }

        double cosineSimilarity = dotProduct * 100 / (Math.sqrt(norm1) * Math.sqrt(norm2));

        return ResponseEntity.ok(cosineSimilarity);
    }

    @PostMapping("/product")
    public List<Document> getProducts(@RequestParam String text) {
        return vectorStore.similaritySearch(
                SearchRequest.
                        builder()
                        .query(text)
                        .topK(2)
                        .build());
    }

    @PostMapping("/ask")
    public String getAnswerUsingRag(@RequestParam String query) {
        return chatClient
                .prompt(query)
                .advisors(new QuestionAnswerAdvisor(vectorStore))
                .call()
                .content();
    }

}