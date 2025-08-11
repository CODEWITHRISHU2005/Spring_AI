package com.CodeWithRishu.Spring_AI.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.http.ResponseEntity;
import org.springframework.util.MimeTypeUtils;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/openai")
public class OpenAIController {

    //private final VectorStore vectorStore;
    //private final EmbeddingModel embeddingModel;
    private final ChatClient chatClient;

    @Value("classpath:/static/pexels-reinis-bruzitis-2153679369-33292126.jpg")
    private Resource sampleImage;

    @Autowired
    public OpenAIController(ChatClient.Builder chatClientBuilder) {
        this.chatClient = chatClientBuilder.build();
    }

    @GetMapping("/{message}")
    public ResponseEntity<String> getAnswer(@PathVariable String message) {
        String response = chatClient.prompt(message)
                .call()
                .content();
        return ResponseEntity.ok(response);
    }


    @PostMapping("/recommend")
    public ResponseEntity<String> recommend(@RequestParam String type, @RequestParam String year, @RequestParam String lang) {

        String promptText = """
                I want to watch a {type} movie tonight with a good rating,
                looking for movies around the year {year}.
                The language I'm looking for is {lang}.
                Suggest one specific movie and tell me the cast and length of the movie.
                
                Your response format must be exactly as follows:
                1. Movie Name
                2. Basic Plot
                3. Cast
                4. Length
                5. IMDB rating
                """;

        String response = chatClient.prompt()
                .user(userSpec -> userSpec.text(promptText)
                        .param("type", type)
                        .param("year", year)
                        .param("lang", lang))
                .call()
                .content();

        return ResponseEntity.ok(response);
    }

    @GetMapping("/image-to-text")
    public String imageToText() {
        return chatClient.prompt()
                .user(u -> {
                    u.text("Please describe this image in detail.");
                    u.media(MimeTypeUtils.IMAGE_JPEG, sampleImage);
                })
                .call()
                .content();
    }

//    @PostMapping("/embedding")
//    public float[] embedding(@RequestParam String text) {
//        return embeddingModel.embed(text);
//    }
//
//    @PostMapping("/similarity")
//    public double getSimilarity(@RequestParam String text1, @RequestParam String text2) {
//        float[] embedding1 = embeddingModel.embed(text1);
//        float[] embedding2 = embeddingModel.embed(text2);
//
//        double dotProduct = 0;
//        double norm1 = 0;
//        double norm2 = 0;
//
//        for (int i = 0; i < embedding1.length; i++) {
//            dotProduct += embedding1[i] * embedding2[i];
//            norm1 += Math.pow(embedding1[i], 2);
//            norm2 += Math.pow(embedding2[i], 2);
//        }
//
//        return dotProduct*100 / (Math.sqrt(norm1) * Math.sqrt(norm2));
//    }
//
//    @PostMapping("/product")
//    public List<Document> getProducts(@RequestParam String text) {
//        return vectorStore.similaritySearch(SearchRequest.builder().query(text).topK(2).build());
//    }
}
