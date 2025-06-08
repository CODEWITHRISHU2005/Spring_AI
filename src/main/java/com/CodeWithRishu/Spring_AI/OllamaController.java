package com.CodeWithRishu.Spring_AI;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/ollama")
@CrossOrigin(origins = "http://localhost:5173")
public class OllamaController {

    private final ChatClient chatClient;

    @Autowired
    private EmbeddingModel embeddingModel;

    public OllamaController(OllamaChatModel chatModel) {
        this.chatClient = ChatClient.create(chatModel);
    }


//    public OlIamaController(ChatClient.Builder builder) {
//        this.chatClient = builder.defaultAdvisors(new MessageChatMemoryAdvisor(new InMemoryChatMemory()))
//                .build();
//
//    }

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
}
