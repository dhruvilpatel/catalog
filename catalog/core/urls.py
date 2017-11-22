from rest_framework import routers
from rest_framework.urlpatterns import format_suffix_patterns
from django.conf.urls import url
from django.views.generic import RedirectView, TemplateView

from . import views

app_name = 'core'

# django rest framework endpoints that can generate JSON / HTML
urlpatterns = format_suffix_patterns([
    url(r'^publication/workflow/email-preview/$', views.EmailPreview.as_view(), name='invite_email_preview'),
    url(r'^publication/workflow/invite/$', views.ContactAuthor.as_view(), name='send_invites'),
    url(r'^publication/update-model-url/(?P<token>[\w:-]+)/$', views.UpdateModelUrlView.as_view(),
        name='update_model_url'),
])

# non django rest framework endpoints for authentication, user dashboard, workflow, and search URLs
urlpatterns += [
    url(r'^contact-us/$', views.ContactFormView.as_view(), name='contact_us'),
    url(r'^$', TemplateView.as_view(template_name='index.html'), name='index'),
    url(r'^dashboard/$', views.DashboardView.as_view(), name='dashboard'),
    url(r'^accounts/profile/$', views.UserProfileView.as_view(), name='user_profile'),
    url(r'^bug-report/$', RedirectView.as_view(url='https://gitreports.com/issue/comses/catalog', permanent=False),
        name='report_issues'),
    url(r'^github/$', RedirectView.as_view(url='https://github.com/comses/catalog', permanent=False), name='github'),
    url(r'^publication/workflow/$', views.CuratorWorkflowView.as_view(), name='curator_workflow'),
    url(r'^search/$', views.CatalogSearchView.as_view(), name='haystack_search'),
    url(r'^search/platform/$', views.PlatformSearchView.as_view(), name="platform_search"),
    url(r'^search/sponsor/$', views.SponsorSearchView.as_view(), name="sponsor_search"),
    url(r'^search/tag/$', views.TagSearchView.as_view(), name="tag_search"),
    url(r'^search/journal/$', views.JournalSearchView.as_view(), name="journal_search"),
    url(r'^search/model-documentation/$', views.ModelDocumentationSearchView.as_view(),
        name="model_documentation_search"),
    url(r'^export/$', views.export_data, name="export_data"),

    # Visualization endpoints
    url(r'^visualization/$', views.VisualizationWorkflowView.as_view(), name="visualization"),
    url(r'^pub-year-distribution/$', views.RelationDetail.as_view(), name="pub-year-distribution"),
    url(r'^pub-year-distribution/(?P<id>\w+)/(?P<name>[\w|\W]+)/$', views.RelationDetail.as_view(),
        name='pub-year-distribution'),
    url(r'^publication-journal-relation/$', views.JournalPublication.as_view(), name="publication-journal-relation"),
    url(r'^publication-sponsor-relation/$', views.SponsorPublication.as_view(), name="publication-sponsor-relation"),
    url(r'^publication-platform-relation/$', views.PlatformPublication.as_view(), name="publication-platform-relation"),
    url(r'^publication-author-relation/$', views.AuthorPublication.as_view(), name="publication-author-relation"),
    url(r'^publication-model-documentation-relation/$', views.ModelDocumentationPublicationRelation.as_view(),
        name="publication-model-documentation-relation"),
    url(r'^publicationlist/(?P<id>\w+)/(?P<name>[\w|\W]+)/(?P<year>[\w|\W]+)/$', views.PublicationListDetail.as_view(),
        name='publicationlist'),
    url(r'^publicationlist/(?P<id>\w*)/(?P<name>[\w|\W]*)/(?P<year>[\w|\W]*)/$', views.PublicationListDetail.as_view(),
        name='publicationlist'),

]
