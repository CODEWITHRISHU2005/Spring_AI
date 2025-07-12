package com.CodeWithRishu.Spring_AI;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.content.Media;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.util.MimeTypeUtils;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.net.MalformedURLException;
import java.util.List;

@RestController
public class OllamaVisionController {

    private static final Logger logger = LoggerFactory.getLogger(OllamaVisionController.class);
    private final ChatModel chatModel;

    @Autowired
    public OllamaVisionController(ChatModel chatModel) {
        this.chatModel = chatModel;
    }

    @GetMapping("/describe-image")
    public ResponseEntity<String> describeImage(
            @RequestParam(value = "prompt", defaultValue = "What do you see in this image?") String prompt,
            @RequestParam(value = "imageUrl") String imageUrl
    ) {
        UrlResource imageResource;
        try {
            imageResource = new UrlResource(imageUrl);
            if (!imageResource.exists() || !imageResource.isReadable()) {
                throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Image not found or not readable");
            }
        } catch (MalformedURLException e) {
            logger.error("Invalid image URL: {}", imageUrl, e);
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Invalid image URL");
        }

        UserMessage userMessage = new UserMessage(
                prompt,
                List.of(new Media(MimeTypeUtils.IMAGE_JPEG, imageResource))
        );

        Prompt chatPrompt = new Prompt(userMessage);

        logger.info("Sending prompt and image to the vision model...");
        ChatResponse response = chatModel.call(chatPrompt);
        logger.info("Received response from vision model.");

        String content = response.getResult().getOutput().getText();
        return ResponseEntity.ok(content);
    }
}