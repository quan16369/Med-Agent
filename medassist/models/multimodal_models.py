"""
Multimodal content models for API communication.
Inspired by Kubrick AI's multimodal message format.
"""

from typing import List, Dict, Union, Literal, Optional
from pydantic import BaseModel, Field
from PIL import Image
import base64
from io import BytesIO


class TextContent(BaseModel):
    """Text content in multimodal message."""
    type: Literal["text"] = "text"
    text: str


class ImageUrlContent(BaseModel):
    """Image content in multimodal message (base64 encoded)."""
    type: Literal["image_url"] = "image_url"
    image_url: Dict[str, str] = Field(
        ..., 
        description="Image URL dict with 'url' key containing data URI"
    )
    
    @classmethod
    def from_base64(cls, base64_data: str, mime_type: str = "image/jpeg"):
        """Create from base64 string.
        
        Args:
            base64_data: Base64 encoded image data (with or without data URI prefix)
            mime_type: MIME type of the image
            
        Returns:
            ImageUrlContent instance
        """
        # Add data URI prefix if not present
        if not base64_data.startswith('data:'):
            base64_data = f"data:{mime_type};base64,{base64_data}"
        return cls(image_url={"url": base64_data})
    
    @classmethod
    def from_image(cls, image: Image.Image, format: str = "JPEG"):
        """Create from PIL Image.
        
        Args:
            image: PIL Image
            format: Image format (JPEG, PNG, etc.)
            
        Returns:
            ImageUrlContent instance
        """
        # Convert RGBA to RGB for JPEG
        if format.upper() == "JPEG" and image.mode == "RGBA":
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        
        buffered = BytesIO()
        image.save(buffered, format=format)
        base64_data = base64.b64encode(buffered.getvalue()).decode()
        
        mime_type = f"image/{format.lower()}"
        return cls.from_base64(base64_data, mime_type)
    
    def to_image(self) -> Image.Image:
        """Convert to PIL Image.
        
        Returns:
            PIL Image
        """
        url = self.image_url["url"]
        # Remove data URI prefix if present
        if url.startswith('data:'):
            url = url.split(',', 1)[1]
        img_data = base64.b64decode(url)
        return Image.open(BytesIO(img_data))


class MultimodalMessage(BaseModel):
    """Multimodal message with text and images.
    
    Compatible with OpenAI Vision API and Groq vision models.
    """
    role: Literal["user", "assistant", "system"] = "user"
    content: List[Union[TextContent, ImageUrlContent]] = Field(
        ...,
        description="List of content items (text and/or images)"
    )
    
    @classmethod
    def from_text(cls, text: str, role: str = "user"):
        """Create text-only message.
        
        Args:
            text: Text content
            role: Message role (user, assistant, system)
            
        Returns:
            MultimodalMessage instance
        """
        return cls(role=role, content=[TextContent(text=text)])
    
    @classmethod
    def from_text_and_image(
        cls, 
        text: str, 
        image: Union[str, Image.Image], 
        role: str = "user"
    ):
        """Create message with text and image.
        
        Args:
            text: Text content
            image: Image (base64 string or PIL Image)
            role: Message role
            
        Returns:
            MultimodalMessage instance
        """
        content = [TextContent(text=text)]
        if isinstance(image, str):
            content.append(ImageUrlContent.from_base64(image))
        else:
            content.append(ImageUrlContent.from_image(image))
        return cls(role=role, content=content)
    
    @classmethod
    def from_text_and_images(
        cls,
        text: str,
        images: List[Union[str, Image.Image]],
        role: str = "user"
    ):
        """Create message with text and multiple images.
        
        Args:
            text: Text content
            images: List of images (base64 strings or PIL Images)
            role: Message role
            
        Returns:
            MultimodalMessage instance
        """
        content = [TextContent(text=text)]
        for image in images:
            if isinstance(image, str):
                content.append(ImageUrlContent.from_base64(image))
            else:
                content.append(ImageUrlContent.from_image(image))
        return cls(role=role, content=content)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for API calls.
        
        Returns:
            Dictionary representation
        """
        return {
            "role": self.role,
            "content": [
                item.model_dump(by_alias=True, exclude_none=True) 
                for item in self.content
            ]
        }


class MedicalImageMetadata(BaseModel):
    """Metadata for medical images."""
    image_type: Optional[str] = Field(None, description="Type of medical image (xray, ct, mri, pathology)")
    body_part: Optional[str] = Field(None, description="Body part in image")
    modality: Optional[str] = Field(None, description="Imaging modality")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    study_date: Optional[str] = Field(None, description="Study date")
    findings: Optional[List[str]] = Field(None, description="Clinical findings")
    
    
class MedicalImageInput(BaseModel):
    """Medical image input with metadata."""
    image: Union[str, bytes] = Field(..., description="Image data (base64 string or bytes)")
    metadata: Optional[MedicalImageMetadata] = None
    
    def to_multimodal_message(self, prompt: str) -> MultimodalMessage:
        """Convert to multimodal message.
        
        Args:
            prompt: Text prompt to accompany image
            
        Returns:
            MultimodalMessage instance
        """
        # Add metadata to prompt if available
        if self.metadata:
            meta_str = f"\n\nImage metadata:\n"
            if self.metadata.image_type:
                meta_str += f"- Type: {self.metadata.image_type}\n"
            if self.metadata.body_part:
                meta_str += f"- Body part: {self.metadata.body_part}\n"
            if self.metadata.findings:
                meta_str += f"- Findings: {', '.join(self.metadata.findings)}\n"
            prompt += meta_str
        
        return MultimodalMessage.from_text_and_image(prompt, self.image)
