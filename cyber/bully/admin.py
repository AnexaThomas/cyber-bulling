from django.contrib import admin
from . models import *
# Register your models here.
class userAdmin(admin.ModelAdmin):
    list_display=('name','email')
    readonly_fields = ['name', 'email']
   
    exclude = ['password','rpwd ']
   
    


admin.site.register(user,userAdmin)
admin.site.register(Image)
admin.site.register(Complaint)
admin.site.register(feed)