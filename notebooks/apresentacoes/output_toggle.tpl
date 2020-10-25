{%- extends 'slides_reveal.tpl' -%}



{% block input_group -%}
<div class="input_hidden">
{{ super() }}
</div>
{% endblock input_group %}

{%- block header -%}
{{ super() }}

<script src="//ajax.googleapis.com/ajax/libs/jquery/1.10.2/jquery.min.js"></script>

<style type="text/css">
//div.output_wrapper {
//  margin-top: 0px;
//}
.input_hidden {
  display: none;
//  margin-top: 5px;
}
</style>

<script>
$(document).ready(function(){
  $(".output_wrapper").dblclick(function(){
      $(this).prev('.input_hidden').slideToggle();
  });
})
</script>
{%- endblock header -%}



{% block body %}

{{ super() }}

<link rel="stylesheet" href="reveal.js/css/theme/simple.css" id="theme">

<script>
require(
    {
      // it makes sense to wait a little bit when you are loading
      // reveal from a cdn in a slow connection environment
      waitSeconds: 1
    },
    [
      "reveal.js/lib/js/head.min.js",
      "reveal.js/js/reveal.js"
    ],

    function(head, Reveal){

        // Full list of configuration options available here: https://github.com/hakimel/reveal.js#configuration
        Reveal.initialize({
            controls: false,
            progress: false,
            history: true,

            theme: Reveal.getQueryHash().theme, // available themes are in /css/theme
            transition: Reveal.getQueryHash().transition || 'linear', // default/cube/page/concave/zoom/linear/none

            // Optional libraries used to extend on reveal.js
            dependencies: [
                { src: "reveal.js/lib/js/classList.js",
                  condition: function() { return !document.body.classList; } },
                { src: "reveal.js/plugin/notes/notes.js",
                  async: true,
                  condition: function() { return !!document.body.classList; } }
            ]
        });

        var update = function(event){
          if(MathJax.Hub.getAllJax(Reveal.getCurrentSlide())){
            MathJax.Hub.Rerender(Reveal.getCurrentSlide());
          }
        };

        Reveal.addEventListener('slidechanged', update);

        var update_scroll = function(event){
          $(".reveal").scrollTop(0);
        };

        Reveal.addEventListener('slidechanged', update_scroll);

    }
);
</script>

{% endblock body %}





