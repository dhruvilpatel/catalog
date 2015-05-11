from rest_framework.urlpatterns import format_suffix_patterns
from django.conf.urls import url
from django.contrib.auth.decorators import login_required
from django.views.generic import RedirectView, TemplateView

from .views import (LoginView, LogoutView, DashboardView, PublicationDetail, PublicationList, EmailPreview,
                    ContactAuthor, ArchivePublication, CustomSearchView, ContactUsView, UserProfileView,
                    PlatformSearchView, SponsorSearchView, TagSearchView, JournalSearchView, ModelDocSearchView,
                    CuratorPublicationDetail, AssignedPubSearchView, NoteList, NoteDetail, )

from .forms import CustomSearchForm

# django rest framework endpoints that can generate JSON / HTML
urlpatterns = format_suffix_patterns([
    url(r'^publications/$', PublicationList.as_view(), name='publications'),
    url(r'^publication/(?P<pk>\d+)/$', PublicationDetail.as_view(), name='publication_detail'),
    url(r'^publication/(?P<pk>\d+)/curate$', CuratorPublicationDetail.as_view(), name='curator_publication_detail'),
    url(r'^publication/workflow/email-preview$', EmailPreview.as_view(), name='invite_email_preview'),
    url(r'^publication/workflow/invite$', ContactAuthor.as_view(), name='send_invites'),
    url(r'^publication/archive/(?P<token>[\w:-]+)$', ArchivePublication.as_view(), name='publication_archive'),
    url(r'^notes/$', NoteList.as_view(), name='notes'),
    url(r'^note/(?P<pk>\d+)$', NoteDetail.as_view(), name='note_detail'),
    url(r'^contact-us/$', ContactUsView.as_view(), name='contact_us'),
    url(r'^account/profile/$', UserProfileView.as_view(), name='user_profile'),
])

urlpatterns += [
    url(r'^$', TemplateView.as_view(template_name='index.html'), name='index'),
    url(r'^dashboard/$', DashboardView.as_view(), name='dashboard'),
    url(r'^account/login/$', LoginView.as_view(), name='login'),
    url(r'^account/logout/$', LogoutView.as_view(), name='logout'),
    url(r'^bug-report/$', RedirectView.as_view(url='https://github.com/comses/catalog/issues/new'),
        name='report_issues')
]
# search endpoints
urlpatterns += [
    url(r'^publication/workflow/$', login_required(AssignedPubSearchView(form_class=CustomSearchForm)),
        name='curator_workflow'),
    url(r'^search/$', login_required(CustomSearchView(form_class=CustomSearchForm)), name='haystack_search'),
    url(r'^search/platform/$', PlatformSearchView.as_view(), name="platform_search"),
    url(r'^search/sponsor/$', SponsorSearchView.as_view(), name="sponsor_search"),
    url(r'^search/tag/$', TagSearchView.as_view(), name="tag_search"),
    url(r'^search/journal/$', JournalSearchView.as_view(), name="journal_search"),
    url(r'^search/modeldoc/$', ModelDocSearchView.as_view(), name="model_doc_search"),
]
