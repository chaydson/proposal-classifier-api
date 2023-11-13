# proposal-classifier-api
API feita usando FastAPI no Ubuntu 22

Para rodar deve-se usar os comandos na pasta raiz:

` sudo docker build -t proposal-classifier .`
` sudo docker run -p 80:80 proposal-classifier `

Então a API será acessível pelo endereço http://0.0.0.0/

Para acessar a documentação swagger deve-se acessar http://0.0.0.0/docs

# Ordem de requests

Primeiro você deve realizar um get request no end-point http://0.0.0.0/ e aguardar o seu carregamento, pois assim o modelo será iniciado.

Após esse primeiro get você poderá usar o end-point http://0.0.0.0/predict que executará as predições do modelo. Segue um exemplo de como usar:

~~~
{
	"text": "proposta de lei que quero prever"
}
~~~

O resultado será um objeto proposal com a predição do modelo:

~~~
{
	"proposal": "Cidades"
}
~~~
