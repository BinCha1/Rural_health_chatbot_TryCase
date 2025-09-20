import os
from django.db import models
from django.conf import settings
from accounts.models import User

def document_upload_path(instance, filename):
    # File will be uploaded to MEDIA_ROOT/documents/<filename>
    return os.path.join('documents', filename)

class Document(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to=document_upload_path)
    summary = models.TextField(blank=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_documents')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title
    
    def get_file_extension(self):
        return os.path.splitext(self.file.name)[1][1:].lower()
    
    def get_file_size(self):
        # Return file size in KB
        return round(self.file.size / 1024, 2)
    
    @property
    def query_count(self):
        # This will be implemented later to count how many times this document was referenced in chats
        from chat.models import ChatHistory
        return ChatHistory.objects.filter(answer__icontains=self.title).count()